import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import mcubes
import point_cloud_utils as pcu
from scipy.ndimage import uniform_filter, median_filter, grey_erosion, binary_erosion
from my.registry import Registry

VOXRF_REGISTRY = Registry("VoxRF")


def to_grid_samp_coords(xyz_sampled, aabb):
    # output range is [-1, 1]
    aabbSize = aabb[1] - aabb[0]
    return (xyz_sampled - aabb[0]) / aabbSize * 2 - 1


def add_non_state_tsr(nn_module, key, val):
    # tsr added here does not appear in module's state_dict;
    nn_module.register_buffer(key, val, persistent=False)


@VOXRF_REGISTRY.register()
class VoxRF(nn.Module):
    def __init__(
        self, aabb, grid_size, step_ratio=0.5,
        density_shift=-10, ray_march_weight_thres=0.0001, c=3,
        blend_bg_texture=True, bg_texture_hw=64
    ):
        assert aabb.shape == (2, 3)
        xyz = grid_size
        del grid_size

        super().__init__()
        add_non_state_tsr(self, "aabb", torch.tensor(aabb, dtype=torch.float32))
        add_non_state_tsr(self, "grid_size", torch.LongTensor(xyz))

        self.density_shift = density_shift
        self.ray_march_weight_thres = ray_march_weight_thres
        self.step_ratio = step_ratio

        zyx = xyz[::-1]
        self.density = torch.nn.Parameter(
            torch.zeros((1, 1, *zyx))
        )
        self.color = torch.nn.Parameter(
            torch.randn((1, c, *zyx))
        )

        self.blend_bg_texture = blend_bg_texture
        self.bg = torch.nn.Parameter(
            torch.randn((1, c, bg_texture_hw, bg_texture_hw))
        )

        self.c = c
        self.alphaMask = None
        self.feats2color = lambda feats: torch.sigmoid(feats)

        self.d_scale = torch.nn.Parameter(torch.tensor(0.0))

        background_latents = torch.load('data/vae_latents.pt').mean(0)
        self.white_bg = background_latents.reshape(4, -1).T

    @property
    def device(self):
        return self.density.device
    
    @torch.no_grad()
    def export_mesh(self, path, reso_mult=2, threshold=None, threshold_mult=8, kernel_size=7, kernel_type="avg", erosion_kernel_size=5):
        if reso_mult is None:
            sigma = self.density
            reso_mult = 1
        else:
            sigma = F.interpolate(self.density, scale_factor=reso_mult, mode="trilinear")
            # raise NotImplementedError
        
        # same as compute_density_feats
        sigma = sigma * torch.exp(self.d_scale)
        sigma = F.softplus(sigma + self.density_shift)

        # smooth
        if kernel_size > 1:
            if kernel_type == "avg":
                sigma = F.avg_pool3d(sigma, kernel_size, 1, kernel_size // 2)
        sigma = sigma.squeeze(0).squeeze(0)
        
        if threshold is None:
            threshold = torch.mean(sigma).item() * threshold_mult
            print("using density threshold:", threshold)

        sigma = sigma.detach().cpu().numpy()
        if erosion_kernel_size > 1:
            sigma = grey_erosion(sigma, size=(erosion_kernel_size, erosion_kernel_size, erosion_kernel_size))
        
        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        # vertices, triangles = mcubes.marching_cubes(sigma > threshold, 0.5)

        # scale vertices to [-0.5, 0.5]
        vertices = vertices / (self.grid_size.detach().cpu().numpy() * reso_mult) - 0.5

        # orientation
        vertices = vertices[:, [2, 0, 1]]
        vertices[:, 2] = -vertices[:, 2]

        pcu.save_mesh_vf(path, vertices, triangles)

    def compute_density_feats(self, xyz_sampled):
        xyz_sampled = to_grid_samp_coords(xyz_sampled, self.aabb)
        n = xyz_sampled.shape[0]
        xyz_sampled = xyz_sampled.reshape(1, n, 1, 1, 3)
        σ = F.grid_sample(self.density, xyz_sampled).view(n)
        # We notice that DreamFusion also uses an exp scaling on densities.
        # The technique here is developed BEFORE DreamFusion came out,
        # and forms part of our upcoming technical report discussing invariant
        # scaling for volume rendering. The research was presented to our
        # funding agency (TRI) on Aug. 25th, and discussed with a few researcher friends
        # during the period.
        σ = σ * torch.exp(self.d_scale)
        σ = F.softplus(σ + self.density_shift)
        return σ

    def compute_app_feats(self, xyz_sampled):
        xyz_sampled = to_grid_samp_coords(xyz_sampled, self.aabb)
        n = xyz_sampled.shape[0]
        xyz_sampled = xyz_sampled.reshape(1, n, 1, 1, 3)
        feats = F.grid_sample(self.color, xyz_sampled).view(self.c, n)
        feats = feats.T
        return feats

    def compute_bg(self, uv):
        n = uv.shape[0]
        uv = uv.reshape(1, n, 1, 2)
        feats = F.grid_sample(self.bg, uv).view(self.c, n)
        feats = feats.T
        return feats

    def get_per_voxel_length(self):
        aabb_size = self.aabb[1] - self.aabb[0]
        # NOTE I am not -1 on grid_size here;
        # I interpret a voxel as a square and val sits at the center; like pixel
        # this is consistent with align_corners=False
        vox_xyz_length = aabb_size / self.grid_size
        return vox_xyz_length

    def get_num_samples(self, max_size=None):
        # funny way to set step size; whatever
        unit = torch.mean(self.get_per_voxel_length())
        step_size = unit * self.step_ratio
        step_size = step_size.item()  # get the float

        if max_size is None:
            aabb_size = self.aabb[1] - self.aabb[0]
            aabb_diag = torch.norm(aabb_size)
            max_size = aabb_diag

        num_samples = int((max_size / step_size).item()) + 1
        return num_samples, step_size

    @torch.no_grad()
    def resample(self, target_xyz: list):
        zyx = target_xyz[::-1]
        self.density = self._resamp_param(self.density, zyx)
        self.color = self._resamp_param(self.color, zyx)
        target_xyz = torch.LongTensor(target_xyz).to(self.aabb.device)
        add_non_state_tsr(self, "grid_size", target_xyz)

    @staticmethod
    def _resamp_param(param, target_size):
        return torch.nn.Parameter(F.interpolate(
            param.data, size=target_size, mode="trilinear"
        ))

    @torch.no_grad()
    def compute_volume_alpha(self):
        xyz = self.grid_size.tolist()
        unit_xyz = self.get_per_voxel_length()
        xs, ys, zs = torch.meshgrid(
            *[torch.arange(nd) for nd in xyz], indexing="ij"
        )
        pts = torch.stack([xs, ys, zs], dim=-1).to(unit_xyz.device)  # [nx, ny, nz, 3]
        pts = self.aabb[0] + (pts + 0.5) * unit_xyz
        pts = pts.reshape(-1, 3)
        # could potentially filter with alpha mask itself if exists
        σ = self.compute_density_feats(pts)
        d = torch.mean(unit_xyz)
        α = 1 - torch.exp(-σ * d)
        α = rearrange(α.view(xyz), "x y z -> 1 1 z y x")
        α = α.contiguous()
        return α

    @torch.no_grad()
    def make_alpha_mask(self):
        α = self.compute_volume_alpha()
        ks = 3
        α = F.max_pool3d(α, kernel_size=ks, padding=ks // 2, stride=1)
        α = (α > 0.08).float()
        vol_mask = AlphaMask(self.aabb, α)
        self.alphaMask = vol_mask

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.alphaMask is not None:
            state['alpha_mask'] = self.alphaMask.export_state()
        return state

    def load_state_dict(self, state_dict):
        if 'alpha_mask' in state_dict.keys():
            state = state_dict.pop("alpha_mask")
            self.alphaMask = AlphaMask.from_state(state)
        return super().load_state_dict(state_dict, strict=True)


@VOXRF_REGISTRY.register()
class V_SJC(VoxRF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # rendering color in [-1, 1] range, since score models all operate on centered img
        self.feats2color = lambda feats: torch.sigmoid(feats) * 2 - 1

    def opt_params(self):
        groups = []
        for name, param in self.named_parameters():
            # print(f"{name} {param.shape}")
            grp = {"params": param}
            if name in ["bg"]:
                grp["lr"] = 0.0001
            if name in ["density"]:
                # grp["lr"] = 0.
                pass
            groups.append(grp)
        return groups

    def annealed_opt_params(self, base_lr, σ):
        groups = []
        for name, param in self.named_parameters():
            # print(f"{name} {param.shape}")
            grp = {"params": param, "lr": base_lr * σ}
            if name in ["density"]:
                grp["lr"] = base_lr * σ
            if name in ["d_scale"]:
                grp["lr"] = 0.
            if name in ["color"]:
                grp["lr"] = base_lr * σ
            if name in ["bg"]:
                grp["lr"] = 0.01
            groups.append(grp)
        return groups


@VOXRF_REGISTRY.register()
class V_SD(V_SJC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # rendering in feature space; no sigmoid thresholding
        self.feats2color = lambda feats: feats


class AlphaMask(nn.Module):
    def __init__(self, aabb, alphas):
        super().__init__()
        zyx = list(alphas.shape[-3:])
        add_non_state_tsr(self, "alphas", alphas.view(1, 1, *zyx))
        xyz = zyx[::-1]
        add_non_state_tsr(self, "grid_size", torch.LongTensor(xyz))
        add_non_state_tsr(self, "aabb", aabb)

    def sample_alpha(self, xyz_pts):
        xyz_pts = to_grid_samp_coords(xyz_pts, self.aabb)
        xyz_pts = xyz_pts.view(1, -1, 1, 1, 3)
        α = F.grid_sample(self.alphas, xyz_pts).view(-1)
        return α

    def export_state(self):
        state = {}
        alphas = self.alphas.bool().cpu().numpy()
        state['shape'] = alphas.shape
        state['mask'] = np.packbits(alphas.reshape(-1))
        state['aabb'] = self.aabb.cpu()
        return state

    @classmethod
    def from_state(cls, state):
        shape = state['shape']
        mask = state['mask']
        aabb = state['aabb']

        length = np.prod(shape)
        alphas = torch.from_numpy(
            np.unpackbits(mask)[:length].reshape(shape)
        )
        amask = cls(aabb, alphas.float())
        return amask


def test():
    device = torch.device("cuda:1")

    aabb = 1.5 * np.array([
        [-1, -1, -1],
        [1, 1, 1]
    ])
    model = VoxRF(aabb, [10, 20, 30])
    model.to(device)
    print(model.density.shape)
    print(model.grid_size)

    return


if __name__ == "__main__":
    test()
