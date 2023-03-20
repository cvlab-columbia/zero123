import math
import fire
import gradio as gr
import numpy as np
import rich
import torch
from contextlib import nullcontext
from einops import rearrange
from functools import partial
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_and_preprocess, instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from torch import autocast
from torchvision import transforms


_SHOW_INTERMEDIATE = True
_GPU_INDEX = 0
# _GPU_INDEX = 2


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location=device)
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale,
                 ddim_eta, x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([math.radians(x), math.sin(
                math.radians(y)), math.cos(math.radians(y)), z])
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            c_concat = model.encode_first_stage((input_im.to(c.device))).mode().detach()
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)
            print(samples_ddim.shape)
            # samples_ddim = torch.nn.functional.interpolate(samples_ddim, 64, mode='nearest', antialias=False)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    model,
    device,
    input_im,
    preprocess=True,
    x=0.,
    y=0.,
    z=0.,
    scale=3.0,
    n_samples=4,
    ddim_steps=50,
    ddim_eta=1.0,
    precision='fp32',
    h=256,
    w=256,
):
    # input_im[input_im == [0., 0., 0.]] = [1., 1., 1., 1.]
    print('old input_im:', input_im.size)

    if preprocess:
        input_im = load_and_preprocess(input_im)
        input_im = (input_im / 255.0).astype(np.float32)
        # (H, W, 3) array in [0, 1].

    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0
        # (H, W, 4) array in [0, 1].
        
        # old method: very important, thresholding background
        # input_im[input_im[:, :, -1] <= 0.9] = [1., 1., 1., 1.]

        # new method: apply correct method of compositing to avoid sudden transitions / thresholding
        # (smoothly transition foreground to white background based on alpha values)
        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im
        
        input_im = input_im[:, :, 0:3]
        # (H, W, 3) array in [0, 1].

    print('new input_im:', input_im.shape, input_im.dtype, input_im.min(), input_im.max())
    show_in_im = Image.fromarray((input_im * 255).astype(np.uint8))

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im * 2 - 1
    input_im = transforms.functional.resize(input_im, [h, w])

    sampler = DDIMSampler(model)
    x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w,
                                  ddim_steps, n_samples, scale, ddim_eta, x, y, z)

    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    
    if _SHOW_INTERMEDIATE:
        return (output_ims, show_in_im)
    else:
        return output_ims


description = '''
Generate novel viewpoints of an object depicted in one input image using a fine-tuned version of Stable Diffusion.
'''

article = '''
## How to use this?
TBD
## How does this work?
TBD
'''


def run_demo(
    device_idx=_GPU_INDEX,
    ckpt='last.ckpt',
    config='configs/sd-objaverse-finetune-c_concat-256.yaml',
):

    device = f'cuda:{device_idx}'
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    inputs = [
        gr.Image(type='pil', image_mode='RGBA'),  # shape=[512, 512]
        gr.Checkbox(True, label='Preprocess image (remove background and center)',
        info='If enabled, the uploaded image will be preprocessed to remove the background and center the object by cropping and/or padding as necessary. '
        'If disabled, the image will be used as-is, *BUT* a fully transparent or white background is required.'),
        # gr.Number(label='polar (between axis z+)'),
        # gr.Number(label='azimuth (between axis x+)'),
        # gr.Number(label='z (distance from center)'),
        gr.Slider(-90, 90, value=0, step=5, label='Polar angle (vertical rotation in degrees)',
        info='Positive values move the camera down, while negative values move the camera up.'),
        gr.Slider(-90, 90, value=0, step=5, label='Azimuth angle (horizontal rotation in degrees)',
        info='Positive values move the camera right, while negative values move the camera left.'),
        gr.Slider(-2, 2, value=0, step=0.5, label='Radius (distance from center)',
        info='Positive values move the camera further away, while negative values move the camera closer.'),
        gr.Slider(0, 30, value=3, step=1, label='cfg scale'),
        gr.Slider(1, 8, value=4, step=1, label='Number of samples to generate'),
        gr.Slider(5, 200, value=100, step=5, label='Number of steps'),
    ]
    output = [gr.Gallery(label='Generated images from specified new viewpoint')]
    output[0].style(grid=2)
    
    if _SHOW_INTERMEDIATE:
        output += [gr.Image(type='pil', image_mode='RGB', label='Preprocessed input image')]

    fn_with_model = partial(main, model, device)
    fn_with_model.__name__ = 'fn_with_model'

    examples = [
        # ['assets/zero-shot/bear.png', 0, 0, 0, 3, 4, 100],
        # ['assets/zero-shot/car.png', 0, 0, 0, 3, 4, 100],
        # ['assets/zero-shot/elephant.png', 0, 0, 0, 3, 4, 100],
        # ['assets/zero-shot/pikachu.png', 0, 0, 0, 3, 4, 100],
        # ['assets/zero-shot/spyro.png', 0, 0, 0, 3, 4, 100],
        # ['assets/zero-shot/taxi.png', 0, 0, 0, 3, 4, 100],
    ]

    demo = gr.Interface(
        fn=fn_with_model,
        title='Demo for Zero-Shot Control of Camera Viewpoints within a Single Image',
        description=description,
        article=article,
        inputs=inputs,
        outputs=output,
        examples=examples,
        allow_flagging='never',
    )
    demo.launch(enable_queue=True, share=True)


if __name__ == '__main__':
    fire.Fire(run_demo)
