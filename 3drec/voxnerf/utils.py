import numpy as np
import math


def blend_rgba(img):
    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
    return img


class PSNR():
    @classmethod
    def psnr(cls, ref, pred, max=1.0):
        # if inputs of type int, then make sure max is 255
        mse = ((ref - pred) ** 2).mean()
        return cls.psnr_from_mse(mse, max)

    @staticmethod
    def psnr_from_mse(mse, max=1.0):
        psnr = 20 * math.log10(max) - 10 * math.log10(mse)
        return psnr

    @staticmethod
    def psnr_to_rms(psnr_diff):
        """rms error improvement _ratio_ from psnr _diff_"""
        ratio = 10 ** (-psnr_diff / 20)
        return ratio


class Scrambler():
    def __init__(self, N):
        self.perm = np.random.permutation(N)

    def apply(self, *items):
        return [elem[self.perm] for elem in items]

    def unscramble(self, *items):
        ret = []
        for elem in items:
            clean = np.zeros_like(elem)
            clean[self.perm] = elem
            ret.append(clean)
        return ret


def trailing_window_view(xs, window_size):
    assert (window_size % 2) == 1, "window size should be odd"
    view = np.lib.stride_tricks.sliding_window_view(
        np.pad(xs, (window_size - 1, 0), mode="edge"), window_size
    )
    return view


def to_step(pbar, percent):
    step = int(pbar.total * percent / 100)
    return step


def every(pbar, *, percent=None, step=None):
    if step is None:
        step = to_step(pbar, percent)
    return (pbar.n + 1) % step == 0


def at(pbar, *, percent=None, step=None):
    if step is None:
        step = to_step(pbar, percent)
    return (pbar.n + 1) == step
