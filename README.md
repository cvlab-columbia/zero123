# Zero-1-to-3: Zero-shot One Image to 3D Object
### [Project Page](https://zero123.cs.columbia.edu/)  | [Paper](https://arxiv.org/abs/2303.11328) | [Weights](https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing) | [Demo (precomputed)](https://huggingface.co/spaces/cvlab/zero123) | [Demo (live)](https://huggingface.co/spaces/cvlab/zero123-live)

<p align="center">
  <img width="90%" src="teaser.png">
</p>

[Zero-1-to-3: Zero-shot One Image to 3D Object](https://zero123.cs.columbia.edu/)  
 [Ruoshi Liu](https://ruoshiliu.github.io/)<sup>1</sup>, [Rundi Wu](https://www.cs.columbia.edu/~rundi/)<sup>1</sup>, [Basile Van Hoorick](https://basile.be/about-me/)<sup>1</sup>, [Pavel Tokmakov](https://pvtokmakov.github.io/home/)<sup>2</sup>, [Sergey Zakharov](https://zakharos.github.io/)<sup>2</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/)<sup>1</sup> <br>
 <sup>1</sup>Columbia University, <sup>2</sup>Toyota Research Institute

## Updates
### $\text{\color{red}{Live demo released}}$ 🤗: https://huggingface.co/spaces/cvlab/zero123-live. Shout out to Huggingface for funding this demo!!
### we've optimized our code base with some simple tricks and the current demo runs at around 22GB VRAM so it's runnable on a RTX 3090(Ti)!

##  Usage
###  Novel View Synthesis
```
conda create -n zero123 python=3.9
conda activate zero123
cd zero123
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download checkpoint under `zero123` through one of the following sources:

```
https://drive.google.com/drive/folders/1geG1IO15nWffJXsmQ_6VLih7ryNivzVs?usp=sharing
https://huggingface.co/cvlab/zero123-weights
wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt
wget https://cv.cs.columbia.edu/zero123/assets/165000.ckpt
```
Note that we have released two model weights. By default, we use 105000.ckpt which is the checkpoint after finetuning 105000 iterations on objaverse. 165000.ckpt is also available.

Run our gradio demo for novel view synthesis:

```
python gradio_new.py
```

Note that this app uses around 22 GB of VRAM, so it may not be possible to run it on any GPU.

### 3D Reconstruction
Note that we haven't extensively tuned the hyperparameters for 3D recosntruction. Feel free to explore and play around!
```
cd 3drec
pip install -r requirements.txt
python run_zero123.py \
    --scene pikachu \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 0 \
    --depth_smooth_weight 10000. \
    --near_view_weight 10000. \
    --train_view True \
    --prefix "experiments/exp_wild" \
    --vox.blend_bg_texture False \
    --nerf_path "data/nerf_wild"
```
We tested the installation processes on a system with Ubuntu 20.04 with an NVIDIA GPU with Ampere architecture.

##  Acknowledgement
This repository is based on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Objaverse](https://objaverse.allenai.org/), and [SJC](https://github.com/pals-ttic/sjc/). We would like to thank the authors of these work for publicly releasing their code. We would like to thank the authors of [NeRDi](https://arxiv.org/abs/2212.03267) and [SJC](https://github.com/pals-ttic/sjc/) for their helpful feedback.

We would like to thank Changxi Zheng for many helpful discussions. This research is based on work partially supported by the Toyota Research Institute, the DARPA MCS program under Federal Agreement No. N660011924032, and the NSF NRI Award #1925157.

##  Citation
```
@misc{liu2023zero1to3,
      title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
      author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
      year={2023},
      eprint={2303.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
