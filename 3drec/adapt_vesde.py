from pathlib import Path
import torch
from ml_collections.config_flags import config_flags

from sde.config import get_config
from sde import ddpm, ncsnv2, ncsnpp  # need to import to trigger its registry
from sde import utils as mutils
from sde.ema import ExponentialMovingAverage

from adapt import ScoreAdapter

device = torch.device("cuda")


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    # state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


class VESDE(ScoreAdapter):
    def __init__(self):
        config = get_config()
        config.device = device
        ckpt_fname = self.checkpoint_root() / "sde" / 'checkpoint_127.pth'

        score_model = mutils.create_model(config)
        ema = ExponentialMovingAverage(
            score_model.parameters(), decay=config.model.ema_rate
        )
        state = dict(model=score_model, ema=ema, step=0)
        self._data_shape = (
            config.data.num_channels, config.data.image_size, config.data.image_size
        )

        self._σ_min = float(config.model.sigma_min * 2)

        state = restore_checkpoint(ckpt_fname, state, device=config.device)
        ema.copy_to(score_model.parameters())

        score_model.eval()
        score_model = score_model.module  # remove DataParallel

        self.model = score_model
        self._device = device

    def data_shape(self):
        return self._data_shape

    @property
    def σ_min(self):
        return self._σ_min

    @torch.no_grad()
    def denoise(self, xs, σ):
        N = xs.shape[0]
        # see Karras eqn. 212-215 for the 1/2 σ correction
        cond_t = (0.5 * σ) * torch.ones(N, device=self.device)
        # note that the forward function the model has been modified; see comments
        n_hat = self.model(xs, cond_t)
        Ds = xs + σ * n_hat
        return Ds

    def unet_is_cond(self):
        return False

    def use_cls_guidance(self):
        return False

    def snap_t_to_nearest_tick(self, t):
        return super().snap_t_to_nearest_tick(t)
