# some tools developed for the vision class
import numpy as np
from numpy import cross, tan
from numpy.linalg import norm, inv
import math, torch


def normalize(v):
    return v / norm(v)

def gradient_x(img: torch.Tensor) -> torch.Tensor:
    return img[:, :-1] - img[:, 1:]

def gradient_y(img: torch.Tensor) -> torch.Tensor:
    return img[:-1, :] - img[1:, :]

def depth_smooth_loss(depth):
    grad_x, grad_y = gradient_x(depth), gradient_y(depth)
    return (grad_x.abs().mean() + grad_y.abs().mean()) / 2.

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])

def get_T(T_target, T_cond):
    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T

def camera_pose(eye, front, up):
    # print('eye', eye)
    # print('front', front)
    # print('up', up)
    z = normalize(-1 * front) # -1 except for mesh
    x = normalize(cross(up, z))
    y = normalize(cross(z, x))
 
    # convert to col vector
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)
    eye = eye.reshape(-1, 1)

    pose = np.block([
        [x, y, z, eye],
        [0, 0, 0, 1]
    ])
    return pose


def compute_extrinsics(eye, front, up):
    pose = camera_pose(eye, front, up)
    world_2_cam = inv(pose)
    return world_2_cam


def compute_intrinsics(aspect_ratio, fov, img_height_in_pix):
    # aspect ratio is  w / h
    ndc = compute_proj_to_normalized(aspect_ratio, fov)

    # anything beyond [-1, 1] should be discarded
    # this did not mention how to do z-clipping;

    ndc_to_img = compute_normalized_to_img_trans(aspect_ratio, img_height_in_pix)
    intrinsic = ndc_to_img @ ndc
    return intrinsic


def compute_proj_to_normalized(aspect, fov):
    # compared to standard OpenGL NDC intrinsic,
    # this skips the 3rd row treatment on z. hence the name partial_ndc
    fov_in_rad = fov / 180 * np.pi
    t = tan(fov_in_rad / 2)  # tan half fov
    partial_ndc_intrinsic = np.array([
        [1 / (t * aspect), 0, 0, 0],
        [0, 1 / t, 0, 0],
        [0, 0, -1, 0]  # copy the negative distance for division
    ])
    return partial_ndc_intrinsic


def compute_normalized_to_img_trans(aspect, img_height_in_pix):
    img_h = img_height_in_pix
    img_w = img_height_in_pix * aspect

    # note the OpenGL convention that (0, 0) sits at the center of the pixel;
    # hence the extra -0.5 translation
    # this is useful when you shoot rays through a pixel to the scene
    ndc_to_img = np.array([
        [img_w / 2, 0, img_w / 2 - 0.5],
        [0, img_h / 2, img_h / 2 - 0.5],
        [0, 0, 1]
    ])

    img_y_coord_flip = np.array([
        [1, 0, 0],
        [0, -1, img_h - 1],  # note the -1
        [0, 0, 1]
    ])

    # the product of the above 2 matrices is equivalent to adding
    # - sign to the (1, 1) entry
    # you could have simply written
    # ndc_to_img = np.array([
    #     [img_w / 2, 0, img_w / 2 - 0.5],
    #     [0, -img_h / 2, img_h / 2 - 0.5],
    #     [0, 0, 1]
    # ])

    ndc_to_img = img_y_coord_flip @ ndc_to_img
    return ndc_to_img


def unproject(K, pixel_coords, depth=1.0):
    """sometimes also referred to as backproject
        pixel_coords: [n, 2] pixel locations
        depth: [n,] or [,] depth value. of a shape that is broadcastable with pix coords
    """
    K = K[0:3, 0:3]

    pixel_coords = as_homogeneous(pixel_coords)
    pixel_coords = pixel_coords.T  # [2+1, n], so that mat mult is on the left

    # this will give points with z = -1, which is exactly what you want since
    # your camera is facing the -ve z axis
    pts = inv(K) @ pixel_coords

    pts = pts * depth  # [3, n] * [n,] broadcast
    pts = pts.T
    pts = as_homogeneous(pts)
    return pts


"""
these two functions are changed so that they can handle arbitrary number of
dimensions >=1
"""


def homogenize(pts):
    # pts: [..., d], where last dim of the d is the diviser
    *front, d = pts.shape
    pts = pts / pts[..., -1].reshape(*front, 1)
    return pts


def as_homogeneous(pts, lib=np):
    # pts: [..., d]
    *front, d = pts.shape
    points = lib.ones((*front, d + 1))
    points[..., :d] = pts
    return points


def simple_point_render(pts, img_w, img_h, fov, eye, front, up):
    """
    pts: [N, 3]
    """
    canvas = np.ones((img_h, img_w, 3))

    pts = as_homogeneous(pts)

    E = compute_extrinsics(eye, front, up)
    world_2_ndc = compute_proj_to_normalized(img_w / img_h, fov)
    ndc_to_img = compute_normalized_to_img_trans(img_w / img_h, img_h)

    pts = pts @ E.T
    pts = pts @ world_2_ndc.T
    pts = homogenize(pts)

    # now filter out outliers beyond [-1, 1]
    outlier_mask = (np.abs(pts) > 1.0).any(axis=1)
    pts = pts[~outlier_mask]

    pts = pts @ ndc_to_img.T

    # now draw each point
    pts = np.rint(pts).astype(np.int32)
    xs, ys, _ = pts.T
    canvas[ys, xs] = (1, 0, 0)

    return canvas
