import numpy as np
from numpy import sin, cos
from math import pi as π
from my3d import camera_pose
from my.config import BaseConf
import random


def get_K(H, W, FoV_x):
    FoV_x = FoV_x / 180 * π  # to rad
    f = 1 / np.tan(FoV_x / 2) * (W / 2)

    K = np.array([
        [f, 0, -(W/2 - 0.5)],
        [0, -f, -(H/2 - 0.5)],
        [0, 0, -1]
    ])
    return K


SIDEVIEW_PROMPTS = [
    "front view of", "side view of", "backside view of", "side view of"
]

TOPVIEW_PROMPT = "overhead view of"

def sample_random_vector(norm=0.05):
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    vec *= norm
    return vec
def sample_near_eye(eye, norm=0.05):
    near_eye = eye + sample_random_vector(norm=norm)
    return near_eye

def sample_point_on_sphere(radius: float):
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def train_eye_with_prompts(r, n, legacy=False):
    if legacy:
        hs = np.random.rand(n) * 360
        vs = np.random.rand(n) * np.deg2rad(100)
        vs = np.clip(vs, 1e-2, π-1e-2)

        prompts = []
        v_thresh = np.deg2rad(30)
        for i in range(n):
            _p = ""
            if vs[i] < v_thresh:
                _p = TOPVIEW_PROMPT
            else:
                _a = hs[i]
                _a = (_a + 45) % 360
                _quad = int(_a // 90)
                _p = SIDEVIEW_PROMPTS[_quad]
            prompts.append(_p)

        θ = np.deg2rad(hs)
        # φ = v
        φ = np.arccos(1 - 2 * (vs / π))

        eyes = np.zeros((n, 3))

        eyes[:, 0] = r * sin(φ) * cos(π-θ)  # x
        eyes[:, 1] = r * sin(φ) * sin(π-θ)  # z
        eyes[:, 2] = r * cos(φ)  # y

    else:
        eyes = np.zeros((n, 3))
        for i in range(n):
            eyes[i] = sample_spherical(radius_min=1.8, radius_max=2.2, maxz=2.2, minz=-0.5)
        prompts = []

    return eyes, prompts


def spiral_poses(
    radius, height,
    num_steps=20, num_rounds=1,
    center=np.array([0, 0, 0]), up=np.array([0, 0, 1]),
):
    eyes = []
    for i in range(num_steps):
        ratio = (i + 1) / num_steps
        Δz = height * (1 - ratio)

        θ = ratio * (360 * num_rounds)
        θ = θ / 180 * π
        # _r = max(radius * ratio, 0.5)
        _r = max(radius * sin(ratio * π / 2), 0.5)
        Δx, Δy = _r * np.array([np.cos(θ), np.sin(θ)])
        eyes.append(center + [Δx, Δy, Δz])

    poses = [
        camera_pose(e, center - e, up) for e in eyes
    ]
    return poses


class PoseConfig(BaseConf):
    rend_hw: int = 64
    FoV: float = 60.0
    R: float = 1.5
    up: str = 'z'

    def make(self):
        cfgs = self.dict()
        hw = cfgs.pop("rend_hw")
        cfgs["H"] = hw
        cfgs["W"] = hw
        return Poser(**cfgs)


class Poser():
    def __init__(self, H, W, FoV, R, up='z'):
        self.H, self.W = H, W
        self.R = R
        self.K = get_K(H, W, FoV)
        if up == 'z':
            self.up = np.array([0, 0, 1])
        elif up == 'y':
            self.up = np.array([0, 1, 0])
        elif up == 'x':
            self.up = np.array([1, 0, 0])

    def sample_train(self, n):
        eyes, prompts = train_eye_with_prompts(r=self.R, n=n)
        up = self.up
        poses = [
            camera_pose(e, -e, up) for e in eyes
        ]
        poses = np.stack(poses, 0)
        random_Ks = [
            self.K for i in range(len(poses)) # objaverse
        ]
        # return self.K, poses, prompts
        return random_Ks, poses, prompts

    def sample_test(self, n):
        print(self.up)
        poses = spiral_poses(self.R, self.R, n, num_rounds=3, up=self.up)
        poses.reverse()
        poses = np.stack(poses, axis=0)
        return self.K, poses

    def get_K(self, H, W, FoV_x):
        return get_K(H, W, FoV_x)