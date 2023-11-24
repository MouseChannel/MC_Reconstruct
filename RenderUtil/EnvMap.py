import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import torch

from RenderUtil import Render_Util, BSDF


class EnvLight:
    def __init__(self):
        return;

    def reflect(self, insert, normal):
        return 2 * torch.dot(-insert, normal) * normal + insert

    def shade(self, v_pos, normals, kd, ks, albedo, view_pos, light_pos, LUT, irradiance_cubemap, skybox):
        roughness = ks[..., 1:2]  # y component
        metallic = ks[..., 2:3]  # z component
        # spec_col = (1.0 - metallic) * 0.04 + kd * metallic
        # diff_col = kd * (1.0 - metallic)
        # wo = Render_Util.safe_normalize(view_pos - v_pos)
        # NdotV = torch.clamp(Render_Util.dot(wo, normals), min=1e-4)
        # fg_uv = torch.cat((NdotV, roughness), dim=-1)
        V = Render_Util.safe_normalize(view_pos - v_pos)
        L = Render_Util.safe_normalize(light_pos - v_pos)
        H = Render_Util.safe_normalize(V + L)

        F0 = torch.tensor(0.04)
        F = BSDF.bsdf_fresnel_shlick(F0, 1, torch.max(torch.dot(V, H), 0.), F0)
        torch.lerp(F0, albedo, metallic)
        KS = F
        KD = torch.tensor(1) - KS
        irradiance = dr.texture(irradiance_cubemap, normals, boundary_mode='cube')
        diffuse = albedo * irradiance
        dir = reflect(-V, normals)
        prefilteredColor = dr.texture(skybox, dir, boundary_mode='cube')
        lut_uv = torch.max(torch.dot(normals, V), 0.)
        lut_uv = roughness
        brdf = dr.texture(LUT, lut_uv, filter_mode='liner')
        
        specular = prefilteredColor * (F0 * brdf.r + brdf.g)
        return KD * diffuse + specular

    @staticmethod
    def to_faces(hdr_path):
        def cube_to_dir(s, x, y):
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

        res = [512, 512]
        image = imageio.imread(hdr_path)
        image = torch.from_numpy(image).cuda().float()
        cubemap = torch.zeros(6, res[0], res[1], 3).cuda()
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'),
                                    torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                    indexing='ij')
            v = Render_Util.safe_normalize(cube_to_dir(s, gx, gy))
            tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
            tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
            texcoord = torch.cat((tu, tv), dim=-1)
            # aopfjsoa = dr.texture(image[None, ...], texcoord[None, ...], filter_mode='linear')

            cubemap[s, ...] = dr.texture(image[None, ...], texcoord[None, ...], filter_mode='linear')[0]
        return cubemap


# context = dr.RasterizeCudaContext();
# context.
# adasd =  torch.nn.functional.normalize(torch.tensor([0.5,0.5,0.5]))
# a = EnvLight.to_faces('/home/mocheng/project/RECONSTRCUT/MC_Reconstruct/aerodynamics_workshop_2k.hdr')
# for i in range(6):
#     temp = a[i]
#     temp = temp.cpu().byte()
#     # temp = temp.astype(np.char) 
#     alpha = torch.ones(temp.shape[0], temp.shape[1], 1).byte() * 255
#     # temp = torch.from_numpy(temp)
#     temp = torch.cat([temp, alpha], -1)
#     imageio.imwrite(str(i) + '.png', temp.numpy())
# sd = 1;

aa = torch.tensor([1, 2, 3])
bb = torch.tensor([1, 2, 3])
aaaa = torch.dot(aa, bb)


def reflect(insert, normal):
    return 2 * torch.dot(-insert, normal) * normal + insert


aaaa = reflect(-torch.tensor([0.5, 1.5, 1.]), torch.tensor([0., 0., 1.]))
qw = 1
