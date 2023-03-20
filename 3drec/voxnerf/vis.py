from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import Normalize, LogNorm
import torch
from torchvision.utils import make_grid
from einops import rearrange
from .data import blend_rgba

import imageio

from my.utils.plot import mpl_fig_to_buffer
from my.utils.event import read_stats


def vis(ref_img, pred_img, pred_depth, *, msg="", return_buffer=False):
    # plt the 2 images side by side and compare
    fig = plt.figure(figsize=(15, 6))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(1, 3),
        cbar_location="right", cbar_mode="single",
    )

    grid[0].imshow(ref_img)
    grid[0].set_title("gt")

    grid[1].imshow(pred_img)
    grid[1].set_title(f"rendering {msg}")

    h = grid[2].imshow(pred_depth, norm=LogNorm(vmin=2, vmax=10), cmap="Spectral")
    grid[2].set_title("expected depth")
    plt.colorbar(h, cax=grid.cbar_axes[0])
    plt.tight_layout()

    if return_buffer:
        plot = mpl_fig_to_buffer(fig)
        return plot
    else:
        plt.show()


def _bad_vis(pred_img, pred_depth, *, return_buffer=False):
    """emergency function for one-off use"""
    fig, grid = plt.subplots(1, 2, squeeze=True, figsize=(10, 6))

    grid[0].imshow(pred_img)
    grid[0].set_title("rendering")

    h = grid[1].imshow(pred_depth, norm=LogNorm(vmin=0.5, vmax=10), cmap="Spectral")
    grid[1].set_title("expected depth")
    # plt.colorbar(h, cax=grid.cbar_axes[0])
    plt.tight_layout()

    if return_buffer:
        plot = mpl_fig_to_buffer(fig)
        return plot
    else:
        plt.show()


colormap = plt.get_cmap('Spectral')


def bad_vis(pred_img, pred_depth, final_H=512):
    # pred_img = pred_img.cpu()
    depth = pred_depth.cpu().numpy()
    del pred_depth

    depth = np.log(1. + depth + 1e-12)
    depth = depth / np.log(1+10.)
    # depth = 1 - depth
    depth = colormap(depth)
    depth = blend_rgba(depth)
    depth = rearrange(depth, "h w c -> 1 c h w", c=3)
    depth = torch.from_numpy(depth)

    depth = torch.nn.functional.interpolate(
        depth, (final_H, final_H), mode='bilinear', antialias=True
    )
    pred_img = torch.nn.functional.interpolate(
        pred_img, (final_H, final_H), mode='bilinear', antialias=True
    )
    pred_img = (pred_img + 1) / 2
    pred_img = pred_img.clamp(0, 1).cpu()
    stacked = torch.cat([pred_img, depth], dim=0)
    pane = make_grid(stacked, nrow=2)
    pane = rearrange(pane, "c h w -> h w c")
    pane = (pane * 255.).clamp(0, 255)
    pane = pane.to(torch.uint8)
    pane = pane.numpy()
    # plt.imshow(pane)
    # plt.show()
    return pane

def vis_img(pred_img, final_H=512):
    pred_img = torch.nn.functional.interpolate(
        pred_img, (final_H, final_H), mode='bilinear', antialias=True
    )
    pred_img = (pred_img + 1) / 2
    pred_img = pred_img.clamp(0, 1).cpu()
    pred_img = rearrange(pred_img, "c h w -> h w c")
    pred_img = (pred_img * 255.).clamp(0, 255)
    pred_img = pred_img.to(torch.uint8)
    pred_img = pred_img.numpy()
    return pred_img


def export_movie(seqs, fname, fps=30):
    fname = Path(fname)
    if fname.suffix == "":
        fname = fname.with_suffix(".mp4")
    writer = imageio.get_writer(fname, fps=fps)
    for img in seqs:
        writer.append_data(img)
    writer.close()


def stitch_vis(save_fn, img_fnames, fps=10):
    figs = [imageio.imread(fn) for fn in img_fnames]
    export_movie(figs, save_fn, fps)
