import nvdiffrast.torch as dr
import torch

from RenderUtil import BSDF
from RenderUtil import Render_Util


# from cvtest import mycvtest


class RenderContext:
    def __init__(self):
        self.context = dr.RasterizeCudaContext()

    def render_single_layer(self, rast, rast_dp, mesh, material, view_pos, light_pos, light_power):
        ###########################
        # Interpolate attributes
        ###########################
        pos, _ = dr.interpolate(mesh.v_pos, rast, mesh.pos_faces)

        # mycvtest.show(pos)
        uv, uv_da = dr.interpolate(mesh.uvs, rast, mesh.uv_faces, rast_dp)
        nrm, _ = dr.interpolate(mesh.nrms, rast, mesh.nrm_faces, )
        tangent, _ = dr.interpolate(mesh.tangent, rast, mesh.tangent_face)

        v0 = mesh.v_pos[mesh.pos_faces[:, 0], :]
        v1 = mesh.v_pos[mesh.pos_faces[:, 1], :]
        v2 = mesh.v_pos[mesh.pos_faces[:, 2], :]
        face_normals = Render_Util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
        face_normal_indices = (
            torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
        gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast, face_normal_indices.int())
        ####################
        # shading
        ####################
        diffuse = material.diffuse_texture.sample(uv, uv_da, filter_mode="linear")
        arm_texture = material.arm.sample(uv, uv_da, filter_mode="linear")

        nrm_texture = material.nrm.sample(uv, uv_da, filter_mode="linear")

        nrm = BSDF.bsdf_prepare_shading_normal(pos, view_pos, nrm_texture, nrm, tangent, gb_geometric_normal,
                                               two_sided_shading=True, opengl=True)

        alpha = diffuse[..., 3:4] if diffuse.shape[-1] == 4 else torch.ones_like(diffuse[..., 0:1])

        diffuse = diffuse[..., 0:3]

        result = BSDF.bsdf_pbr(diffuse, arm_texture, pos, nrm, view_pos, light_pos) * light_power
        return torch.cat((result, alpha), dim=-1)

    def rasterize(self, gameobject, mvp, width, height, view_pos, light_pos, light_power, num_layers=1):
        # clip space transform for v_pos
        mesh = gameobject.mesh
        fixed_mesh = gameobject.center_by_reference(mesh, mesh.aabb())
        points = fixed_mesh.v_pos[None, ...]

        v_pos_clip = torch.matmul(torch.nn.functional.pad(points,
                                                          pad=(0, 1),
                                                          mode='constant',
                                                          value=1.0),
                                  torch.transpose(mvp, 1, 2))

        with dr.DepthPeeler(self.context,
                            v_pos_clip,
                            fixed_mesh.pos_faces.int(),
                            [width, height]) as peeler:
            data = torch.tensor([],device='cuda')
            for _ in range(num_layers):
                rast, rast_dp = peeler.rasterize_next_layer()
                data = torch.cat([data, self.render_single_layer(rast,
                                                                 rast_dp,
                                                                 fixed_mesh,
                                                                 gameobject.material,
                                                                 view_pos,
                                                                 light_pos,
                                                                 light_power)], 0)
            return data


render_context = RenderContext()
