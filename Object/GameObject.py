import copy

import torch

from Mesh import Mesh
from Material import Material
from RenderUtil import *
from RenderUtil.RenderContext import render_context


class Gameobject:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.mesh.compute_tangents()
        self.material = material

    def center_by_reference(self, base_mesh, ref_aabb, scale=1.):
        '''
        move mesh to center of screen,
        :param ref_aabb: 
        :param scale: 
        :return: 
        '''
        center = (ref_aabb[0] + ref_aabb[1]) * 0.5
        scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item()
        v_pos = (base_mesh.v_pos - center[None, ...]) * scale
        new_mesh = copy.copy(base_mesh)
        new_mesh.v_pos = v_pos
        return new_mesh

    def render(self, mvp, view_pos, light_pos, light_power):
        def prepare_data(data):
            if len(data.shape) == 2:
                return data[:, None, None, :]
            else:
                return data

        return render_context.rasterize(self, mvp, 1024, 1024, prepare_data(view_pos), prepare_data(light_pos),
                                        prepare_data(light_power))

# a_mv =  util.lookAt(eye, at, up)
# a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]
