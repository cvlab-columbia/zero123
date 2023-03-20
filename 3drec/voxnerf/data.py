from pathlib import Path
import json
import numpy as np
import imageio
import os
import cv2
from .utils import blend_rgba


def load_blender(split, scene="lego", half_res=False, path="data/nerf_synthetic"):
    assert split in ("train", "val", "test")

    root = Path(path) / scene

    with open(root / f'transforms_{split}.json', "r") as f:
        meta = json.load(f)

    imgs, poses = [], []

    for frame in meta['frames']:
        file_name = root / f"{frame['file_path']}.png"
        im = imageio.imread(file_name)
        im = cv2.resize(im, (800, 800), interpolation = cv2.INTER_CUBIC)

        c2w = frame['transform_matrix']

        imgs.append(im)
        poses.append(c2w)

    imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = imgs[:, :, :, -1]
    imgs = blend_rgba(imgs)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    f = 1 / np.tan(camera_angle_x / 2) * (W / 2)

    if half_res:
        raise NotImplementedError()

    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])  # note OpenGL -ve z convention;

    fov = meta['camera_angle_x']

    return imgs, K, poses, mask, fov

def load_wild(dataset_root, scene, index):

    root = Path(dataset_root) / scene

    with open(root / f'transforms_train.json', "r") as f:
        meta = json.load(f)
        
    frame = meta['frames'][index]
    file_name = root / f"{frame['file_path']}.png"
    img = imageio.imread(file_name)

    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
    img = (np.array(img) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = img[:, :, -1]
    img = blend_rgba(img)

    pose = meta['frames'][index]['transform_matrix']
    pose = np.array(pose)

    return img, pose, mask, None


def load_googlescan_data(dataset_root, scene, index, split='render_mvs'):
    render_folder = os.path.join(dataset_root, scene, split, "model")
    if not os.path.exists(render_folder):
        print(f"Render folder {render_folder} does not exist")

    image_path = os.path.join(render_folder, f"{index:03d}.png")
    cam_path = os.path.join(render_folder, f"{index:03d}.npy")
    mesh_path = os.path.join(dataset_root, scene, split, "model_norm.obj")
    if not os.path.exists(mesh_path):
        mesh_path = None

    img = imageio.imread(image_path)
    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
    img = (np.array(img) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = img[:, :, -1]
    img = blend_rgba(img)

    pose = np.load(cam_path) # [3, 4]
    if pose.shape == (3, 4):
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0) # [4, 4]
    return img, pose, mask, mesh_path


def load_rtmv_data(dataset_root, scene, index):
    scene_dir = os.path.join(dataset_root, scene)
    if not os.path.exists(scene_dir):
        print(f"Render folder {scene_dir} does not exist")

    image_path = os.path.join(scene_dir, f"{index:05d}.png")
    mesh_path = os.path.join(dataset_root, scene, "scene.ply")
    if not os.path.exists(mesh_path):
        mesh_path = None

    img = imageio.imread(image_path)
    img = cv2.resize(img, (800, 800), interpolation = cv2.INTER_CUBIC)
    img = (np.array(img) / 255.).astype(np.float32)  # (RGBA) imgs
    mask = img[:, :, -1]
    img = blend_rgba(img)

    with open(os.path.join(scene_dir, "transforms.json"), "r") as f:
        meta = json.load(f)

    pose = meta['frames'][index]['transform_matrix']
    pose = np.array(pose)
    return img, pose, mask, mesh_path
