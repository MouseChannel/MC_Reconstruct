import numpy as np
import torch


def projection(fov=60., aspect=1., near=1.0, far=1000.0
               ):
    wei = 3.14 / 180
    tan_half = np.tan(wei * fov / 2.)
    ######## [1][1] set negative to make it correct😒
    return np.array([[1. / (aspect * tan_half), 0, 0, 0],
                     [0,                        -1. / tan_half, 0, 0],
                     [0,                           0, -(far + near) / (far - near), -(2 * far * near) / (far - near)],
                     [0,                            0, -1, 0]]).astype(np.float32)


def lookAt(eye, at, up):
    """    
    :param eye: 
    :param at: 
    :param up: 
    :return: M_V matrix 
    """
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate = np.array([[1, 0, 0, -eye[0]],
                          [0, 1, 0, -eye[1]],
                          [0, 0, 1, -eye[2]],
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate = np.array([[u[0], u[1], u[2], 0],
                       [v[0], v[1], v[2], 0],
                       [w[0], w[1], w[2], 0],
                       [0, 0, 0, 1]]).astype(np.float32)
    return np.matmul(rotate, translate)

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)
# def get_normal_space(nrm):
#     someVec = torch.tensor([1.0, 0.0, 0.0]);
#     dd = torch.dot(someVec, nrm);
#     tangent = torch.tensor([0.0, 1.0, 0.0]);
#     if 1.0 - abs(dd > 1e-6):
#         tangent = torch.nn.functional.normalize(torch.multiply(someVec, nrm))
# 
#          
#         bitangent = torch.multiply(nrm, tangent);
#     return mat3(tangent, bitangent, normal);



def scale_img_nhwc(x  : torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC