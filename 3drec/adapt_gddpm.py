from pathlib import Path
from math import sin, pi, sqrt
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,

    NUM_CLASSES,
    create_classifier,
    classifier_defaults,

    sr_create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
)

from adapt import ScoreAdapter

from my.registry import Registry

PRETRAINED_REGISTRY = Registry("pretrained")


device = torch.device("cuda")


def load_ckpt(path, **kwargs):
    # with bf.BlobFile(path, "rb") as f:
    #     data = f.read()
    return torch.load(path, **kwargs)


def pick_out_cfgs(src, target_ks):
    return {k: src[k] for k in target_ks}


@PRETRAINED_REGISTRY.register()
def m_imgnet_64():
    return dict(
        attention_resolutions="32,16,8",
        class_cond=True,
        diffusion_steps=1000,
        dropout=0.1,
        image_size=64,
        learn_sigma=True,
        noise_schedule="cosine",
        num_channels=192,
        num_head_channels=64,
        num_res_blocks=3,
        resblock_updown=True,
        use_new_attention_order=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        classifier_depth=4,

        classifier_scale=1.0,
        model_path="models/64x64_diffusion.pt",
        classifier_path="models/64x64_classifier.pt",
    )


@PRETRAINED_REGISTRY.register()
def m_imgnet_128():
    return dict(
        attention_resolutions="32,16,8",
        class_cond=True,
        diffusion_steps=1000,
        image_size=128,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_heads=4,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        classifier_scale=0.5,
        model_path="models/128x128_diffusion.pt",
        classifier_path="models/128x128_classifier.pt",
    )


@PRETRAINED_REGISTRY.register()
def m_imgnet_256():
    return dict(
        attention_resolutions="32,16,8",
        class_cond=True,
        diffusion_steps=1000,
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        classifier_scale=1.0,
        model_path="models/256x256_diffusion.pt",
        classifier_path="models/256x256_classifier.pt"
    )


@PRETRAINED_REGISTRY.register()
def m_imgnet_256_uncond():
    return dict(
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        classifier_scale=10.0,
        model_path="models/256x256_diffusion_uncond.pt",
        classifier_path="models/256x256_classifier.pt",
    )


@PRETRAINED_REGISTRY.register()
def m_imgnet_512():
    return dict(
        attention_resolutions="32,16,8",
        class_cond=True,
        diffusion_steps=1000,
        image_size=512,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=False,
        use_scale_shift_norm=True,

        classifier_scale=4.0,
        model_path="models/512x512_diffusion.pt",
        classifier_path="models/512x512_classifier.pt"
    )


@PRETRAINED_REGISTRY.register()
def m_imgnet_64_256(base_samples="64_samples.npz"):
    return dict(
        attention_resolutions="32,16,8",
        class_cond=True,
        diffusion_steps=1000,
        large_size=256,
        small_size=64,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=192,
        num_heads=4,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        model_path="models/64_256_upsampler.pt",

        base_samples=base_samples,
    )


@PRETRAINED_REGISTRY.register()
def m_imgnet_128_512(base_samples="128_samples.npz",):
    return dict(
        attention_resolutions="32,16",
        class_cond=True,
        diffusion_steps=1000,
        large_size=512,
        small_size=128,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=192,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        model_path="models/128_512_upsampler.pt",

        base_samples=base_samples,
    )


@PRETRAINED_REGISTRY.register()
def m_lsun_256(category="bedroom"):
    return dict(
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        dropout=0.1,
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,

        model_path=f"models/lsun_{category}.pt"
    )


def img_gen(specific_cfgs, num_samples=16, batch_size=16, load_only=False, ckpt_root=Path("")):
    cfgs = EasyDict(
        clip_denoised=True,
        num_samples=num_samples,
        batch_size=batch_size,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    cfgs.update(model_and_diffusion_defaults())
    cfgs.update(classifier_defaults())
    cfgs.update(specific_cfgs)

    use_classifier_guidance = bool(cfgs.classifier_path)
    class_aware = cfgs.class_cond or use_classifier_guidance

    model, diffusion = create_model_and_diffusion(
        **pick_out_cfgs(cfgs, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        load_ckpt(str(ckpt_root / cfgs.model_path), map_location="cpu")
    )
    model.to(device)
    if cfgs.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None):
        return model(x, t, y if cfgs.class_cond else None)

    classifier = None
    cond_fn = None
    if use_classifier_guidance:
        classifier = create_classifier(
            **pick_out_cfgs(cfgs, classifier_defaults().keys())
        )
        classifier.load_state_dict(
            load_ckpt(str(ckpt_root / cfgs.classifier_path), map_location="cpu")
        )
        classifier.to(device)
        if cfgs.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return torch.autograd.grad(selected.sum(), x_in)[0] * cfgs.classifier_scale

    if load_only:
        return model, classifier

    all_images = []
    all_labels = []

    while len(all_images) * cfgs.batch_size < cfgs.num_samples:
        model_kwargs = {}

        if class_aware:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(cfgs.batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not cfgs.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (cfgs.batch_size, 3, cfgs.image_size, cfgs.image_size),
            clip_denoised=cfgs.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=True
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.append(sample.cpu().numpy())
        if class_aware:
            all_labels.append(classes.cpu().numpy())

    arr = np.concatenate(all_images, axis=0)
    arr = arr[:cfgs.num_samples]

    if class_aware:
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels[:cfgs.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = Path("./out") / f"samples_{shape_str}.npz"
    np.savez(out_path, arr, all_labels)


def img_upsamp(specific_cfgs, num_samples=16, batch_size=16, load_only=False):
    """note that here the ckpt root is not configured properly; will break but easy fix"""
    cfgs = EasyDict(
        clip_denoised=True,
        num_samples=num_samples,
        batch_size=batch_size,
        use_ddim=False,
        base_samples="",
        model_path="",
    )
    cfgs.update(sr_model_and_diffusion_defaults())
    cfgs.update(specific_cfgs)

    model, diffusion = sr_create_model_and_diffusion(
        **pick_out_cfgs(cfgs, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(load_ckpt(cfgs.model_path, map_location="cpu"))
    model.to(device)
    if cfgs.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if load_only:
        return model

    data = load_low_res_samples(
        cfgs.base_samples, cfgs.batch_size, cfgs.class_cond
    )

    all_images = []
    while len(all_images) * cfgs.batch_size < cfgs.num_samples:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}
        samples = diffusion.p_sample_loop(
            model,
            (cfgs.batch_size, 3, cfgs.large_size, cfgs.large_size),
            clip_denoised=cfgs.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True
        )
        samples = ((samples + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        samples = samples.permute(0, 2, 3, 1)
        samples = samples.contiguous()

        all_images.append(samples.cpu().numpy())

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: cfgs.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = Path("./out") / f"samples_{shape_str}.npz"
    np.savez(out_path, arr)


def load_low_res_samples(base_samples, batch_size, class_cond):
    obj = np.load(base_samples)
    image_arr = obj["arr_0"]
    if class_cond:
        label_arr = obj["arr_1"]

    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])

            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = {}
                res["low_res"] = batch
                if class_cond:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []


def class_cond_info(imgnet_cat):

    def rand_cond_fn(batch_size):
        cats = torch.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=device
        )
        return {"y": cats}

    def class_specific_cond(batch_size):
        cats = torch.tensor([imgnet_cat, ] * batch_size, device=device)
        return {"y": cats}

    if imgnet_cat == -1:
        return rand_cond_fn
    else:
        return class_specific_cond


def _sqrt(x):
    if isinstance(x, float):
        return sqrt(x)
    else:
        assert isinstance(x, torch.Tensor)
        return torch.sqrt(x)


class GuidedDDPM(ScoreAdapter):
    def __init__(self, model, lsun_cat, imgnet_cat):
        print(PRETRAINED_REGISTRY)
        cfgs = PRETRAINED_REGISTRY.get(model)(
            **({"category": lsun_cat} if model.startswith("m_lsun") else {})
        )

        self.unet, self.classifier = img_gen(
            cfgs, load_only=True, ckpt_root=self.checkpoint_root() / "guided_ddpm"
        )

        H, W = cfgs['image_size'], cfgs['image_size']
        self._data_shape = (3, H, W)

        if cfgs['class_cond'] or (self.classifier is not None):
            cond_func = class_cond_info(imgnet_cat)
        else:
            cond_func = lambda *args, **kwargs: {}
        self.cond_func = cond_func

        self._unet_is_cond = bool(cfgs['class_cond'])

        noise_schedule = cfgs['noise_schedule']
        assert noise_schedule in ("linear", "cosine")
        self.M = 1000
        if noise_schedule == "linear":
            self.us = self.linear_us(self.M)
            self._σ_min = 0.01
        else:
            self.us = self.cosine_us(self.M)
            self._σ_min = 0.0064
        self.noise_schedule = noise_schedule

        self._device = next(self.unet.parameters()).device

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
        N = xs.shape[0]
        cond_t, σ = self.time_cond_vec(N, σ)
        output = self.unet(
            xs / _sqrt(1 + σ**2), cond_t, **model_kwargs
        )
        # not using the var pred
        n_hat = torch.split(output, xs.shape[1], dim=1)[0]
        Ds = xs - σ * n_hat
        return Ds

    def cond_info(self, batch_size):
        return self.cond_func(batch_size)

    def unet_is_cond(self):
        return self._unet_is_cond

    def use_cls_guidance(self):
        return (self.classifier is not None)

    @torch.no_grad()
    def classifier_grad(self, xs, σ, ys):
        N = xs.shape[0]
        cond_t, σ = self.time_cond_vec(N, σ)
        with torch.enable_grad():
            x_in = xs.detach().requires_grad_(True)
            logits = self.classifier(x_in, cond_t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), ys.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0]

        grad = grad * (1 / sqrt(1 + σ**2))
        return grad

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
    def cosine_us(M=1000):
        assert M == 1000

        def α_bar(j):
            return sin(pi / 2 * j / (M * (0.008 + 1))) ** 2

        us = [0, ]
        for j in reversed(range(0, M)):  # [M-1, 0], inclusive
            u_j = sqrt(((us[-1] ** 2) + 1) / (max(α_bar(j) / α_bar(j+1), 0.001)) - 1)
            us.append(u_j)

        us = np.array(us)
        us = us[1:]
        us = us[::-1]
        return us

    @staticmethod
    def linear_us(M=1000):
        assert M == 1000
        β_start = 0.0001
        β_end = 0.02
        βs = np.linspace(β_start, β_end, M, dtype=np.float64)
        αs = np.cumprod(1 - βs)
        us = np.sqrt((1 - αs) / αs)
        us = us[::-1]
        return us
