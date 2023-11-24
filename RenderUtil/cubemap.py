import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import torch

from RenderUtil import Render_Util


class Cubemap:
    def __init__(self, hdr_path):

        res = [512, 512]
        image = imageio.imread(hdr_path)
        image = torch.from_numpy(image).cuda().float()
        cubemap = torch.zeros(6, res[0], res[1], 3).cuda()
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
                                    torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                    indexing='ij')
            v = Render_Util.safe_normalize(self.cube_to_dir(s, gx, gy))
            tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
            tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
            texcoord = torch.cat((tu, tv), dim=-1)
            # aopfjsoa = dr.texture(image[None, ...], texcoord[None, ...], filter_mode='linear')

            cubemap[s, ...] = dr.texture(image[None, ...], texcoord[None, ...], filter_mode='linear')[0]
        self.data = cubemap

    def cube_to_dir(self, s, x, y):
        if s == 0:
            rx, ry, rz = torch.ones_like(x), -y, -x
        elif s == 1:
            rx, ry, rz = -torch.ones_like(x), -y, x
        elif s == 2:
            rx, ry, rz = x, torch.ones_like(x), y
        elif s == 3:
            rx, ry, rz = x, -torch.ones_like(x), -y
        elif s == 4:
            rx, ry, rz = x, -y, torch.ones_like(x)
        elif s == 5:
            rx, ry, rz = -x, -y, -torch.ones_like(x)
        return torch.stack((rx, ry, rz), dim=-1)

    def sample(self, uv):
        '''
        uv need to be **vec3**
        :param uv: 
        :return: 
        '''
        return dr.texture(self.data, uv, filter_mode='liner', boundary_mode='cube')

a = Cubemap('/home/mocheng/project/RECONSTRCUT/MC_Reconstruct/aerodynamics_workshop_2k.hdr')
for i in range(6):
    temp = a.data[i]
    temp = temp.cpu().byte()
    # temp = temp.astype(np.char) 
    alpha = torch.ones(temp.shape[0], temp.shape[1], 1).byte() * 255
    # temp = torch.from_numpy(temp)
    temp = torch.cat([temp, alpha], -1)
    imageio.imwrite(str(i) + '.png', temp.numpy())
sd = 1;
