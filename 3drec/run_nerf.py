from typing import List
from pydantic import validator

from my.config import BaseConf, SingleOrList, dispatch
from my.utils.seed import seed_everything

import numpy as np
from voxnerf.vox import VOXRF_REGISTRY
from voxnerf.pipelines import train


class VoxConfig(BaseConf):
    model_type:                 str = "VoxRF"
    bbox_len:                   float = 1.5
    grid_size:                  SingleOrList(int) = [128, 128, 128]
    step_ratio:                 float = 0.5
    density_shift:              float = -10.
    ray_march_weight_thres:     float = 0.0001
    c:                          int = 3
    blend_bg_texture:           bool = False
    bg_texture_hw:              int = 64

    @validator("grid_size")
    def check_gsize(cls, grid_size):
        if isinstance(grid_size, int):
            return [grid_size, ] * 3
        else:
            assert len(grid_size) == 3
            return grid_size

    def make(self):
        params = self.dict()
        m_type = params.pop("model_type")
        model_fn = VOXRF_REGISTRY.get(m_type)

        radius = params.pop('bbox_len')
        aabb = radius * np.array([
            [-1, -1, -1],
            [1, 1, 1]
        ])
        model = model_fn(aabb=aabb, **params)
        return model


class TrainerConfig(BaseConf):
    model:      VoxConfig = VoxConfig()
    scene:      str = "lego"
    n_epoch:    int = 2
    bs:         int = 4096
    lr:         float = 0.02

    def run(self):
        args = self.dict()
        args.pop("model")

        model = self.model.make()
        train(model, **args)


if __name__ == "__main__":
    seed_everything(0)
    dispatch(TrainerConfig)
