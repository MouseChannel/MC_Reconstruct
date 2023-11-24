import torch
import numpy as np

from RenderUtil import Render_Util


def from_azimuth_to_axe(radius, phi, theta):
    '''
           Y 
           |
           |
           |
           |-------------X
          /
    #    / 
    #   /
    #  /
    # /
     Z
    :param phi: 方位角 (0 -- 2PI)
    :param theta: 仰角(0---PI)
    :return: 坐标 (x,y,z)
    '''
    phi = torch.deg2rad(torch.tensor(phi))
    theta = torch.deg2rad(torch.tensor(theta))
    x = torch.cos(phi) * radius
    z = torch.sin(phi) * radius
    y = torch.cos(theta) * radius

    return np.array([x, y, z])


def get_random_camera_pos(radius):
    phi = np.random.randint(0, 360)
    theta = np.random.randint(0, 180)
    return from_azimuth_to_axe(radius, phi, theta)


def get_random_camera_mvp(radius, eye_pos):
    up = np.array([0.0, 1.0, 0.0])
    at = np.array([0, 0, 0])
    a_mv = Render_Util.lookAt(eye_pos, at, up)
    proj = Render_Util.projection()
    return np.matmul(proj, a_mv).astype(np.float32)[None, ...]



a = torch.tensor([])
aaa = torch.ones(2,3)
qw = torch.cat([a,aaa],0)
qwq = torch
www = 1