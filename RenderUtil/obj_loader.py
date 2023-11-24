# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import imageio.v2
import numpy as np
import torch
from Material import Util, Material, Texture2D
from Mesh import Mesh
from Object import GameObject


# import Render_Util

# from cvtest import mycvtest


# from . import util
# from . import texture
# from . import mesh
# from . import material

######################################################################################
# Utility functions
######################################################################################

def _write_weights(folder, mesh):
    if mesh.v_weights is not None:
        file = os.path.join(folder, 'mesh.weights')
        np.save(file, mesh.v_weights.detach().cpu().numpy())


def _write_bones(folder, mesh):
    if mesh.bone_mtx is not None:
        file = os.path.join(folder, 'mesh.bones')
        np.save(file, mesh.bone_mtx.detach().cpu().numpy())


def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0]  # Materials 0 is the default


######################################################################################
# Create mesh object from objfile
######################################################################################


def load_obj(filename):
    mesh = load_obj_mesh(filename)
    material = load_obj_material(filename)

    return GameObject.Gameobject(mesh, material)


def load_obj_mesh(filename):
    with open(filename) as f:
        lines = f.readlines()
    # load vertices
    vertices, texcoords, normals = [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'v':
            vertices.append([float(v) for v in line.split()[1:]])
        elif prefix == 'vt':
            val = [float(v) for v in line.split()[1:]]
            texcoords.append([val[0], 1.0 - val[1]])
        elif prefix == 'vn':
            normals.append([float(v) for v in line.split()[1:]])

    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    for line in lines:
        if len(line.split()) == 0:
            continue

        prefix = line.split()[0].lower()
        if prefix == 'usemtl':  # Track used materials
            # mat = _find_mat(materials, line.split()[1])
            # if not mat in used_materials:
            #     used_materials.append(mat)
            # activeMatIdx = used_materials.index(mat)
            continue
        elif prefix == 'f':  # Parse face
            vs = line.split()[1:]
            nv = len(vs)
            vv = vs[0].split('/')
            v0 = int(vv[0]) - 1
            t0 = int(vv[1]) - 1 if vv[1] != "" else -1
            n0 = int(vv[2]) - 1 if vv[2] != "" else -1
            for i in range(nv - 2):  # Triangulate polygons
                vv = vs[i + 1].split('/')
                v1 = int(vv[0]) - 1
                t1 = int(vv[1]) - 1 if vv[1] != "" else -1
                n1 = int(vv[2]) - 1 if vv[2] != "" else -1
                vv = vs[i + 2].split('/')
                v2 = int(vv[0]) - 1
                t2 = int(vv[1]) - 1 if vv[1] != "" else -1
                n2 = int(vv[2]) - 1 if vv[2] != "" else -1
                mfaces.append(activeMatIdx)
                faces.append([v0, v1, v2])
                tfaces.append([t0, t1, t2])
                nfaces.append([n0, n1, n2])
    assert len(tfaces) == len(faces) and len(nfaces) == len(faces)

    # Create an "uber" material by combining all textures into a larger texture
    # if len(used_materials) > 1:
    #     uber_material, texcoords, tfaces = material.merge_materials(used_materials, texcoords, tfaces, mfaces)
    # else:
    #     uber_material = used_materials[0]

    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None

    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None
    return Mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces)


def load_obj_material(filename):
    obj_path = os.path.dirname(filename)

    with open(filename) as f:
        lines = f.readlines()
    # # Load materials
    # all_materials = [
    #     {
    #         'name' : '_default_mat',
    #         'bsdf' : 'falcor',
    #         'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
    #         'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
    #     }
    # ]

    materials = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'mtllib':
            materials = Util.load_mtl(os.path.join(obj_path, line.split()[1]))  # Read in entire material library
            break
        # else:
        #     raise RuntimeError('no mtllib_file in obj')
    # else:
    #     all_materials += material.load_mtl(mtl_override)
    if len(materials) == 0:
        print(filename, 'has no material')
        return None
    else:
        print(filename, 'has material')
        material = materials[0]
        return Material.Material(material['kd'], material['ks'], material['normal'])
