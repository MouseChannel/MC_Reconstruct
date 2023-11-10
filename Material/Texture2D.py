import os

import imageio
import numpy as np
import nvdiffrast.torch as dr
import torch


class Texture2D:
    def __init__(self, data):
        self.data = data

    def sample(self, uv, uv_da, filter_mode="linear-mipmap-linear"):
        data = self.data
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        return dr.texture(data, uv, uv_da, filter_mode=filter_mode)


def load_image(fn) -> np.ndarray:
    img = imageio.v2.imread(fn)
    if img.dtype == np.float32:  # HDR image
        return img
    else:  # LDR image
        return img.astype(np.float32) / 255


def _load_mip2D(fn, lambda_fn=None, channels=None):
    imgdata = torch.tensor(load_image(fn), dtype=torch.float32, device='cuda')
    if channels is not None:
        imgdata = imgdata[..., 0:channels]
    if lambda_fn is not None:
        imgdata = lambda_fn(imgdata)
    return imgdata.detach().clone()


def load_texture2D(fn, lambda_fn=None, channels=None):
    base, ext = os.path.splitext(fn)
    if os.path.exists(base + "_0" + ext):
        mips = []
        while os.path.exists(base + ("_%d" % len(mips)) + ext):
            mips += [_load_mip2D(base + ("_%d" % len(mips)) + ext, lambda_fn, channels)]
        return Texture2D(mips)
    else:
        return Texture2D(_load_mip2D(fn, lambda_fn, channels))


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
        return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def tonemap_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055, 12.92 * f)


# def create_trainable_Texture():
#     with
