import torch
import numpy as np
from Mesh import Mesh


class DMTET:
    def __init__(self):
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
            [4, 0, 3, -1, -1, -1],
            [1, 4, 2, 1, 3, 4],
            [3, 1, 5, -1, -1, -1],
            [2, 3, 0, 2, 5, 3],
            [1, 4, 0, 1, 5, 4],
            [4, 2, 5, -1, -1, -1],
            [4, 5, 2, -1, -1, -1],
            [4, 1, 0, 4, 5, 1],
            [3, 2, 0, 3, 5, 2],
            [1, 3, 5, -1, -1, -1],
            [4, 1, 2, 4, 3, 1],
            [3, 0, 4, -1, -1, -1],
            [2, 0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long,
                                                device='cuda')
        self.base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device='cuda')

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device="cuda")
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1] * 2,
            torch.stack((tet_gidx[num_triangles == 2] * 2, tet_gidx[num_triangles == 2] * 2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets * 2)

        return verts, faces, uvs, uv_idx


class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS = FLAGS
        self.grid_res = grid_res
        self.marching_tets = DMTET()

        tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        self.verts = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        #  generate_edges
        with torch.no_grad():
            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cuda")
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

        # Random init
        sdf = torch.rand_like(self.verts[:, 0]) - 0.1

        self.sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
        imesh = Mesh.Mesh(verts, faces, uvs, uv_idx, material=material)

        # Run mesh operations to generate tangent space
        # imesh = Mesh.auto_normals()
        # imesh = Mesh.compute_tangents(imesh)

        return imesh
