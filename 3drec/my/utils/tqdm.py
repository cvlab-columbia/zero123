import os
from tqdm import tqdm as orig_tqdm


def tqdm(*args, **kwargs):
    is_remote = bool(os.environ.get("IS_REMOTE", False))
    if is_remote:
        f = open(os.devnull, "w")
        kwargs.update({"file": f})
    return orig_tqdm(*args, **kwargs)
