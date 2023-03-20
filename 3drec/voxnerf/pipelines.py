import numpy as np
import torch
import imageio

from my.utils.tqdm import tqdm
from my.utils.event import EventStorage, read_stats, get_event_storage
from my.utils.heartbeat import HeartBeat, get_heartbeat
from my.utils.debug import EarlyLoopBreak

from .utils import PSNR, Scrambler, every, at
from .data import load_blender
from .render import (
    as_torch_tsrs, scene_box_filter, render_ray_bundle, render_one_view, rays_from_img
)
from .vis import vis, stitch_vis


device_glb = torch.device("cuda")


def all_train_rays(scene):
    imgs, K, poses = load_blender("train", scene)
    num_imgs = len(imgs)
    ro, rd, rgbs = [], [], []
    for i in tqdm(range(num_imgs)):
        img, pose = imgs[i], poses[i]
        H, W = img.shape[:2]
        _ro, _rd = rays_from_img(H, W, K, pose)
        ro.append(_ro)
        rd.append(_rd)
        rgbs.append(img.reshape(-1, 3))

    ro, rd, rgbs = [
        np.concatenate(xs, axis=0) for xs in (ro, rd, rgbs)
    ]
    return ro, rd, rgbs


class OneTestView():
    def __init__(self, scene):
        imgs, K, poses = load_blender("test", scene)
        self.imgs, self.K, self.poses = imgs, K, poses
        self.i = 0

    def render(self, model):
        i = self.i
        img, K, pose = self.imgs[i], self.K, self.poses[i]
        with torch.no_grad():
            aabb = model.aabb.T.cpu().numpy()
            H, W = img.shape[:2]
            rgbs, depth = render_one_view(model, aabb, H, W, K, pose)
            psnr = PSNR.psnr(img, rgbs)

        self.i = (self.i + 1) % len(self.imgs)

        return img, rgbs, depth, psnr


def train(
    model, n_epoch=2, bs=4096, lr=0.02, scene="lego"
):
    fuse = EarlyLoopBreak(500)

    aabb = model.aabb.T.numpy()
    model = model.to(device_glb)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    test_view = OneTestView(scene)
    all_ro, all_rd, all_rgbs = all_train_rays(scene)

    with tqdm(total=(n_epoch * len(all_ro) // bs)) as pbar, \
            HeartBeat(pbar) as hbeat, EventStorage() as metric:

        ro, rd, t_min, t_max, intsct_inds = scene_box_filter(all_ro, all_rd, aabb)
        rgbs = all_rgbs[intsct_inds]

        for epc in range(n_epoch):
            n = len(ro)
            scrambler = Scrambler(n)
            ro, rd, t_min, t_max, rgbs = scrambler.apply(ro, rd, t_min, t_max, rgbs)

            num_batch = int(np.ceil(n / bs))
            for i in range(num_batch):
                if fuse.on_break():
                    break

                s = i * bs
                e = min(n, s + bs)

                optim.zero_grad()
                _ro, _rd, _t_min, _t_max, _rgbs = as_torch_tsrs(
                    model.device, ro[s:e], rd[s:e], t_min[s:e], t_max[s:e], rgbs[s:e]
                )
                pred, _, _ = render_ray_bundle(model, _ro, _rd, _t_min, _t_max)
                loss = ((pred - _rgbs) ** 2).mean()
                loss.backward()
                optim.step()

                pbar.update()

                psnr = PSNR.psnr_from_mse(loss.item())
                metric.put_scalars(psnr=psnr, d_scale=model.d_scale.item())

                if every(pbar, step=50):
                    pbar.set_description(f"TRAIN: psnr {psnr:.2f}")

                if every(pbar, percent=1):
                    gimg, rimg, depth, psnr = test_view.render(model)
                    pane = vis(
                        gimg, rimg, depth,
                        msg=f"psnr: {psnr:.2f}", return_buffer=True
                    )
                    metric.put_artifact(
                        "vis", ".png", lambda fn: imageio.imwrite(fn, pane)
                    )

                if at(pbar, percent=30):
                    model.make_alpha_mask()

                if every(pbar, percent=35):
                    target_xyz = (model.grid_size * 1.328).int().tolist()
                    model.resample(target_xyz)
                    optim = torch.optim.Adam(model.parameters(), lr=lr)
                    print(f"resamp the voxel to {model.grid_size}")

                curr_lr = update_lr(pbar, optim, lr)
                metric.put_scalars(lr=curr_lr)

                metric.step()
                hbeat.beat()

        metric.put_artifact(
            "ckpt", ".pt", lambda fn: torch.save(model.state_dict(), fn)
        )
        # metric.step(flush=True)  # no need to flush since the test routine directly takes the model

        metric.put_artifact(
            "train_seq", ".mp4",
            lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "vis")[1])
        )

        with EventStorage("test"):
            final_psnr = test(model, scene)
        metric.put("test_psnr", final_psnr)

        metric.step()

        hbeat.done()


def update_lr(pbar, optimizer, init_lr):
    i, N = pbar.n, pbar.total
    factor = 0.1 ** (1 / N)
    lr = init_lr * (factor ** i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def last_ckpt():
    ts, ckpts = read_stats("./", "ckpt")
    if len(ckpts) > 0:
        fname = ckpts[-1]
        last = torch.load(fname, map_location="cpu")
        print(f"loaded ckpt from iter {ts[-1]}")
        return last


def __evaluate_ckpt(model, scene):
    # this is for external script that needs to evaluate an checkpoint
    # currently not used
    metric = get_event_storage()

    state = last_ckpt()
    if state is not None:
        model.load_state_dict(state)
    model.to(device_glb)

    with EventStorage("test"):
        final_psnr = test(model, scene)
    metric.put("test_psnr", final_psnr)


def test(model, scene):
    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()
    hbeat = get_heartbeat()

    aabb = model.aabb.T.cpu().numpy()
    model = model.to(device_glb)

    imgs, K, poses = load_blender("test", scene)
    num_imgs = len(imgs)

    stats = []

    for i in (pbar := tqdm(range(num_imgs))):
        if fuse.on_break():
            break

        img, pose = imgs[i], poses[i]
        H, W = img.shape[:2]
        rgbs, depth = render_one_view(model, aabb, H, W, K, pose)
        psnr = PSNR.psnr(img, rgbs)

        stats.append(psnr)
        metric.put_scalars(psnr=psnr)
        pbar.set_description(f"TEST: mean psnr {np.mean(stats):.2f}")

        plot = vis(img, rgbs, depth, msg=f"PSNR: {psnr:.2f}", return_buffer=True)
        metric.put_artifact("test_vis", ".png", lambda fn: imageio.imwrite(fn, plot))
        metric.step()
        hbeat.beat()

    metric.put_artifact(
        "test_seq", ".mp4",
        lambda fn: stitch_vis(fn, read_stats(metric.output_dir, "test_vis")[1])
    )

    final_psnr = np.mean(stats)
    metric.put("final_psnr", final_psnr)
    metric.step()

    return final_psnr
