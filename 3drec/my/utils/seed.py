# from pytorch lightning
import random
import numpy as np
import torch

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def seed_everything(seed=None):
    seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        raise ValueError(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")

    print(f"seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
