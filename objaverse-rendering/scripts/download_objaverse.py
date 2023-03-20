import argparse
import json
import random
from dataclasses import dataclass

import boto3
import objaverse
import tyro
from tqdm import tqdm


@dataclass
class Args:
    start_i: int
    """total number of files uploaded"""

    end_i: int
    """total number of files uploaded"""

    skip_completed: bool = False
    """whether to skip the files that have already been downloaded"""


def get_completed_uids():
    # get all the files in the objaverse-images bucket
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("objaverse-images")
    bucket_files = [obj.key for obj in tqdm(bucket.objects.all())]

    dir_counts = {}
    for file in bucket_files:
        d = file.split("/")[0]
        dir_counts[d] = dir_counts.get(d, 0) + 1

    # get the directories with 12 files
    dirs = [d for d, c in dir_counts.items() if c == 12]
    return set(dirs)


# set the random seed to 42
if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(42)

    uids = objaverse.load_uids()

    random.shuffle(uids)

    object_paths = objaverse._load_object_paths()
    uids = uids[args.start_i : args.end_i]

    # get the uids that have already been downloaded
    if not args.skip_completed:
        completed_uids = get_completed_uids()
        uids = [uid for uid in uids if uid not in completed_uids]

    uid_object_paths = [
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}"
        for uid in uids
    ]

    with open("input_models_path.json", "w") as f:
        json.dump(uid_object_paths, f, indent=2)
