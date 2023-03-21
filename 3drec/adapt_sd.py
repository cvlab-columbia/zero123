import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
from einops import rearrange

from torch import autocast
from torchvision import transforms
from contextlib import nullcontext
from math import sqrt
from adapt import ScoreAdapter
from PIL import Image
import requests
import sys
sys.path.append('../zero123')

import warnings
from transformers import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity_error()


device = torch.device("cuda")


def curr_dir():
    return Path(__file__).resolve().parent


def add_import_path(dirname):
    sys.path.append(str(
        curr_dir() / str(dirname)
    ))

def load_model_from_config(config, ckpt, verbose=False):
    from ldm.util import instantiate_from_config
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

def load_objaverse_model(ckpt_root):
    ckpt_fname = '../zero123/105000.ckpt'
    cfg_fname = '../zero123/configs/sd-objaverse-finetune-c_concat-256.yaml'
    H, W = 256, 256

    config = OmegaConf.load(str(cfg_fname))
    model = load_model_from_config(config, str(ckpt_fname))
    return model, H, W

def _sqrt(x):
    if isinstance(x, float):
        return sqrt(x)
    else:
        assert isinstance(x, torch.Tensor)
        return torch.sqrt(x)


class StableDiffusion(ScoreAdapter):
    def __init__(self, variant, v2_highres, prompt, scale, precision, im_path=None):
        if variant == "objaverse":
            add_import_path("zero123")
            self.model, H, W = load_objaverse_model(self.checkpoint_root())
        else:
            raise ValueError(f"{variant}")

        ae_resolution_f = 8

        self._device = self.model._device

        self.prompt = prompt
        self.im_path = im_path
        self.scale = scale
        self.precision = precision
        self.precision_scope = autocast if self.precision == "autocast" else nullcontext
        self._data_shape = (4, H // ae_resolution_f, W // ae_resolution_f)

        self.cond_func = self.model.get_learned_conditioning
        self.M = 1000
        noise_schedule = "linear"
        self.noise_schedule = noise_schedule
        self.us = self.linear_us(self.M)
        self.clip_emb = None
        self.vae_emb = None

    def data_shape(self):
        return self._data_shape

    @property
    def σ_max(self):
        return self.us[0]

    @property
    def σ_min(self):
        return self.us[-1]

    @torch.no_grad()
    def denoise(self, xs, σ, **model_kwargs):
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                N = xs.shape[0]
                c = model_kwargs.pop('c')
                uc = model_kwargs.pop('uc')
                cond_t, σ = self.time_cond_vec(N, σ)
                unscaled_xs = xs
                xs = xs / _sqrt(1 + σ**2)
                if uc is None or self.scale == 1.:
                    output = self.model.apply_model(xs, cond_t, c)
                else:
                    x_in = torch.cat([xs] * 2)
                    t_in = torch.cat([cond_t] * 2)
                    c_in = torch.cat([uc, c])
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                    output = e_t_uncond + self.scale * (e_t - e_t_uncond)

                if self.model.parameterization == "v":
                    output = self.model.predict_eps_from_z_and_v(xs, cond_t, output)
                else:
                    output = output

                Ds = unscaled_xs - σ * output
                return Ds

    @torch.no_grad()
    def denoise_objaverse(self, xs, σ, score_cond):
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                N = xs.shape[0]
                cond_t, σ = self.time_cond_vec(N, σ)
                unscaled_xs = xs
                xs = xs / _sqrt(1 + σ**2)

                x_in = torch.cat([xs] * 2)
                t_in = torch.cat([cond_t] * 2)
                c_in = score_cond
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                output = e_t_uncond + self.scale * (e_t - e_t_uncond)

                if self.model.parameterization == "v":
                    output = self.model.predict_eps_from_z_and_v(xs, cond_t, output)
                else:
                    output = output

                Ds = unscaled_xs - σ * output
                return Ds

    def cond_info(self, batch_size):
        prompts = batch_size * [self.prompt]
        return self.prompts_emb(prompts)

    @torch.no_grad()
    def prompts_emb(self, prompts):
        assert isinstance(prompts, list)
        batch_size = len(prompts)
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                cond = {}
                c = self.cond_func(prompts)
                cond['c'] = c
                uc = None
                if self.scale != 1.0:
                    uc = self.cond_func(batch_size * [""])
                cond['uc'] = uc
                return cond

    @torch.no_grad()
    def img_emb(self, input_im, conditioning_key='hybrid', T=None):
        if conditioning_key == 'hybrid':
            assert T is not None, 'for objaverse, needs to input T (viewpoint vector)'
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    cond = {}
                    clip_emb = self.model.cc_projection(torch.cat([self.clip_emb, T[None, None, :]], dim=-1))

                    cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
                    cond['c_concat'] = [torch.cat([torch.zeros_like(self.vae_emb).to(self.device), self.vae_emb], dim=0)]
                    # cond['c'] = {'c_crossattn' : [clip_emb],\
                    #              'c_concat' : [self.vae_emb]}
                    # cond['uc'] = {'c_crossattn' : [torch.zeros_like(clip_emb)],\
                    #              'c_concat' : [torch.zeros_like(self.vae_emb)]}
                    return cond
        else:
            with self.precision_scope("cuda"):
                with self.model.ema_scope():
                    cond = {}
                    c = self.model.get_learned_conditioning(input_im).tile(1,1,1)
                    cond['c'] = c
                    cond['uc'] = torch.zeros_like(c)
                    return cond

    def unet_is_cond(self):
        return True

    def use_cls_guidance(self):
        return False

    def snap_t_to_nearest_tick(self, t):
        j = np.abs(t - self.us).argmin()
        return self.us[j], j

    def time_cond_vec(self, N, σ):
        if isinstance(σ, float):
            σ, j = self.snap_t_to_nearest_tick(σ)  # σ might change due to snapping
            cond_t = (self.M - 1) - j
            cond_t = torch.tensor([cond_t] * N, device=self.device)
            return cond_t, σ
        else:
            assert isinstance(σ, torch.Tensor)
            σ = σ.reshape(-1).cpu().numpy()
            σs = []
            js = []
            for elem in σ:
                _σ, _j = self.snap_t_to_nearest_tick(elem)
                σs.append(_σ)
                js.append((self.M - 1) - _j)

            cond_t = torch.tensor(js, device=self.device)
            σs = torch.tensor(σs, device=self.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
            return cond_t, σs

    @staticmethod
    def linear_us(M=1000):
        assert M == 1000
        β_start = 0.00085
        β_end = 0.0120
        βs = np.linspace(β_start**0.5, β_end**0.5, M, dtype=np.float64)**2
        αs = np.cumprod(1 - βs)
        us = np.sqrt((1 - αs) / αs)
        us = us[::-1]
        return us

    # @torch.no_grad()
    def encode(self, xs):
        model = self.model
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                zs = model.get_first_stage_encoding(
                    model.encode_first_stage(xs)
                )
        return zs

    # @torch.no_grad()
    def decode(self, xs):
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                xs = self.model.decode_first_stage(xs)
                return xs


def test():
    sd = StableDiffusion("v2", True, "haha", 10.0, "autocast")
    print(sd)


if __name__ == "__main__":
    test()
