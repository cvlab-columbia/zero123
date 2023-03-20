from pathlib import Path
import json
from math import sqrt
import numpy as np
import torch
import os
from abc import ABCMeta, abstractmethod


class ScoreAdapter(metaclass=ABCMeta):

    @abstractmethod
    def denoise(self, xs, σ, **kwargs):
        pass

    def score(self, xs, σ, **kwargs):
        Ds = self.denoise(xs, σ, **kwargs)
        grad_log_p_t = (Ds - xs) / (σ ** 2)
        return grad_log_p_t

    @abstractmethod
    def data_shape(self):
        return (3, 256, 256)  # for example

    def samps_centered(self):
        # if centered, samples expected to be in range [-1, 1], else [0, 1]
        return True

    @property
    @abstractmethod
    def σ_max(self):
        pass

    @property
    @abstractmethod
    def σ_min(self):
        pass

    def cond_info(self, batch_size):
        return {}

    @abstractmethod
    def unet_is_cond(self):
        return False

    @abstractmethod
    def use_cls_guidance(self):
        return False  # most models do not use cls guidance

    def classifier_grad(self, xs, σ, ys):
        raise NotImplementedError()

    @abstractmethod
    def snap_t_to_nearest_tick(self, t):
        # need to confirm for each model; continuous time model doesn't need this
        return t, None

    @property
    def device(self):
        return self._device

    def checkpoint_root(self):
        """the path at which the pretrained checkpoints are stored"""
        root = os.path.join(os.getcwd(), 'data/release/diffusion_ckpts')
        return root


def karras_t_schedule(ρ=7, N=10, σ_max=80, σ_min=0.002):
    ts = []
    for i in range(N):

        t = (
            σ_max ** (1 / ρ) + (i / (N - 1)) * (σ_min ** (1 / ρ) - σ_max ** (1 / ρ))
        ) ** ρ
        ts.append(t)
    return ts


def power_schedule(σ_max, σ_min, num_stages):
    σs = np.exp(np.linspace(np.log(σ_max), np.log(σ_min), num_stages))
    return σs


class Karras():

    @classmethod
    @torch.no_grad()
    def inference(
        cls, model, batch_size, num_t, *,
        σ_max=80, cls_scaling=1,
        init_xs=None, heun=True,
        langevin=False,
        S_churn=80, S_min=0.05, S_max=50, S_noise=1.003,
    ):
        σ_max = min(σ_max, model.σ_max)
        σ_min = model.σ_min
        ts = karras_t_schedule(ρ=7, N=num_t, σ_max=σ_max, σ_min=σ_min)
        assert len(ts) == num_t
        ts = [model.snap_t_to_nearest_tick(t)[0] for t in ts]
        ts.append(0)  # 0 is the destination
        σ_max = ts[0]

        cond_inputs = model.cond_info(batch_size)

        def compute_step(xs, σ):
            grad_log_p_t = model.score(
                xs, σ, **(cond_inputs if model.unet_is_cond() else {})
            )
            if model.use_cls_guidance():
                grad_cls = model.classifier_grad(xs, σ, cond_inputs["y"])
                grad_cls = grad_cls * cls_scaling
                grad_log_p_t += grad_cls
            d_i = -1 * σ * grad_log_p_t
            return d_i

        if init_xs is not None:
            xs = init_xs.to(model.device)
        else:
            xs = σ_max * torch.randn(
                batch_size, *model.data_shape(), device=model.device
            )

        yield xs

        for i in range(num_t):
            t_i = ts[i]

            if langevin and (S_min < t_i and t_i < S_max):
                xs, t_i = cls.noise_backward_in_time(
                    model, xs, t_i, S_noise, S_churn / num_t
                )

            Δt = ts[i+1] - t_i

            d_1 = compute_step(xs, σ=t_i)
            xs_1 = xs + Δt * d_1

            # Heun's 2nd order method; don't apply on the last step
            if (not heun) or (ts[i+1] == 0):
                xs = xs_1
            else:
                d_2 = compute_step(xs_1, σ=ts[i+1])
                xs = xs + Δt * (d_1 + d_2) / 2

            yield xs

    @staticmethod
    def noise_backward_in_time(model, xs, t_i, S_noise, S_churn_i):
        n = S_noise * torch.randn_like(xs)
        γ_i = min(sqrt(2)-1, S_churn_i)
        t_i_hat = t_i * (1 + γ_i)
        t_i_hat = model.snap_t_to_nearest_tick(t_i_hat)[0]
        xs = xs + n * sqrt(t_i_hat ** 2 - t_i ** 2)
        return xs, t_i_hat


def test():
    pass


if __name__ == "__main__":
    test()
