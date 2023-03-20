from pathlib import Path
import numpy as np
import torch

from misc import torch_samps_to_imgs
from adapt import Karras, ScoreAdapter, power_schedule
# from adapt_vesde import VESDE  # not included to prevent import conflicts
from adapt_sd import StableDiffusion

from my.utils import tqdm, EventStorage, HeartBeat, EarlyLoopBreak
from my.config import BaseConf, dispatch
from my.utils.seed import seed_everything

class SD(BaseConf):
    """Stable Diffusion"""
    variant:        str = "objaverse"
    v2_highres:     bool = False
    prompt:         str = "a photograph of an astronaut riding a horse"
    im_path:        str = "data/nerf_synthetic/chair/train/r_2.png"
    scale:          float = 3.0  # classifier free guidance scale
    precision:      str = 'autocast'

    def make(self):
        args = self.dict()
        model = StableDiffusion(**args)
        return model


def smld_inference(model, σs, num_steps, ε, init_xs):
    from math import sqrt
    # not doing conditioning or cls guidance; for gddpm only lsun works; fine.

    xs = init_xs
    yield xs

    for i in range(len(σs)):
        α_i = ε * ((σs[i] / σs[-1]) ** 2)
        for _ in range(num_steps):
            grad = model.score(xs, σs[i])
            z = torch.randn_like(xs)
            xs = xs + α_i * grad + sqrt(2 * α_i) * z
            yield xs


def load_np_imgs(fname):
    fname = Path(fname)
    data = np.load(fname)
    if fname.suffix == ".npz":
        imgs = data['arr_0']
    else:
        imgs = data
    return imgs


def visualize(max_n_imgs=16):
    import torchvision.utils as vutils
    from imageio import imwrite
    from einops import rearrange

    all_imgs = load_np_imgs("imgs/step_0.npy")

    imgs = all_imgs[:max_n_imgs]
    imgs = rearrange(imgs, "N H W C -> N C H W", C=3)
    imgs = torch.from_numpy(imgs)
    pane = vutils.make_grid(imgs, padding=2, nrow=4)
    pane = rearrange(pane, "C H W -> H W C", C=3)
    pane = pane.numpy()
    imwrite("preview.jpg", pane)


if __name__ == "__main__":
    seed_everything(0)
    dispatch(KarrasGen)
    visualize(16)
