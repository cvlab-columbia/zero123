import objaverse
from tqdm import tqdm
import multiprocessing
processes = multiprocessing.cpu_count()
print(objaverse.__version__)

uids = objaverse.load_uids()
len(uids), type(uids)


annotations = objaverse.load_annotations()

objects = objaverse.load_objects(
    uids=uids[:100],
    download_processes=processes
)