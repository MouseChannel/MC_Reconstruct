import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import torch

from RenderUtil import Render_Util, BSDF


class EnvLight:
    LUT = None
    irradiance_cubemap = None
    skybox = None
    def __init__(self):
        return;

    @staticmethod
    def reflect(insert, normal):

        return 2 * EnvLight.mat_dot(-insert, normal) * normal + insert

    @staticmethod
    def name(s):
        if s == 0:
            return "right"
        elif s == 1:
            return "left"
        elif s == 2:
            return "top"
        elif s == 3:
            return "bottom"
        elif s == 4:
            return "forward"
        elif s == 5:
            return "backward"

    @staticmethod
    def read_skybox(path):
        res = torch.Tensor()
        for i in range(6):
            cur_name = EnvLight.name(i)

            cur = imageio.imread(path + '/' + cur_name + '.png')
            cur = torch.from_numpy(cur)
            res = torch.cat([res, cur[None, ...]], 0)
        return res.cuda()

    @staticmethod
    def mat_dot(a, b):
        return (a * b).sum(-1).unsqueeze(-1)

    @staticmethod
    def shade(v_pos, normals, ks, albedo, view_pos, light_pos, LUT=None, irradiance_cubemap=None, skybox=None):
        if EnvLight.LUT is None:
            EnvLight.LUT = imageio.imread('rainforest_trail_4k_LUT.png')
            EnvLight.LUT = torch.from_numpy(EnvLight.LUT).float().cuda()
        if EnvLight.irradiance_cubemap is None:
            EnvLight.irradiance_cubemap = EnvLight.read_skybox(
                '/home/mocheng/project/RECONSTRCUT/MC_Reconstruct/cubemap/irradiance')
            if EnvLight.irradiance_cubemap.shape[-1] == 4:
                EnvLight.irradiance_cubemap = EnvLight.irradiance_cubemap[..., :-1]
        if EnvLight.skybox is None:
            EnvLight.skybox = EnvLight.read_skybox('/home/mocheng/project/RECONSTRCUT/MC_Reconstruct/cubemap/skybox')

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
        V = V.squeeze(0)
        H = H.squeeze(0)

        cosTheta = EnvLight.mat_dot(V, H)

        F0 = torch.tensor(0.04).cuda()
        F0 = torch.lerp(F0, albedo, metallic)
        F = BSDF.bsdf_fresnel_shlick(F0, 1, cosTheta)
        KS = F
        KD = torch.tensor(1) - KS

        irradiance = dr.texture(EnvLight.irradiance_cubemap[None, ...].contiguous(),
                                normals.contiguous(),
                                filter_mode='linear',
                                boundary_mode='cube')
        diffuse = albedo * irradiance
        diffuse = albedo
        dir = EnvLight.reflect(-V, normals)
        prefilteredColor = dr.texture(EnvLight.skybox[None, ...], dir, filter_mode='linear', boundary_mode='cube')
        # lut_uv =  torch.max(torch.dot(normals, V), 0.)
        # lut_uv = EnvLight.mat_dot(normals, V)
        # 
        # lut_uv = roughness
        lut_uv = torch.cat([EnvLight.mat_dot(normals, V),roughness],-1)
        brdf = dr.texture(EnvLight.LUT[None,...], lut_uv, filter_mode='linear')
        brdf = brdf /255
        
        # F0 * 
        specular = prefilteredColor * (F0 * brdf[...,:1] + brdf[...,1:2])
        return KD * diffuse + specular

    @staticmethod
    def to_faces(hdr_path):
        def cube_to_dir(s, x, y):
            if s == 0:
                # right
                rx, ry, rz = -torch.ones_like(x), -y, -x
            elif s == 1:
                # left
                rx, ry, rz = torch.ones_like(x), -y, x
            elif s == 2:
                # top
                rx, ry, rz = -x, torch.ones_like(x), y
            elif s == 3:
                # bottom
                rx, ry, rz = -x, -torch.ones_like(x), -y
            elif s == 4:
                # forward
                rx, ry, rz = -x, -y, torch.ones_like(x)
            elif s == 5:
                # backward
                rx, ry, rz = x, -y, -torch.ones_like(x)

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

    @staticmethod
    def render_gameobject(gameobject, mvp, width, height, view_pos):
        mesh = gameobject.mesh
        fixed_mesh = gameobject.center_by_reference(mesh, mesh.aabb())
        points = fixed_mesh.v_pos[None, ...]

        v_pos_clip = torch.matmul(torch.nn.functional.pad(points,
                                                          pad=(0, 1),
                                                          mode='constant',
                                                          value=1.0),
                                  torch.transpose(mvp, 1, 2))

        EnvLight.shade(v_pos_clip)

# cube = EnvLight.to_faces('/home/mocheng/project/RECONSTRCUT/MC_Reconstruct/rainforest_trail_4k.hdr')
# for i in range(6):
#     tmp = cube[i].cpu()
#     # tmp = torch.cat([tmp, torch.ones(512, 512, 1)*255], -1).numpy()
#     tmp = tmp.numpy()
#     imageio.imwrite("/home/mocheng/project/RECONSTRCUT/MC_Reconstruct/cubemap/skybox/" + EnvLight.name(i) + '.png',
#                     tmp.astype(np.uint8))
