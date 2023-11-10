import torch

from RenderUtil import Render_Util


class Mesh:
    def __init__(self, v_pos, pos_faces, nrms, nrm_faces, uvs, uv_faces, tangent=None, tangent_face=None,
                 other_mesh=None):

        self.v_pos = v_pos
        self.pos_faces = pos_faces.long()
        self.nrms = nrms
        self.nrm_faces = nrm_faces.long()

        self.uvs = uvs
        self.uv_faces = uv_faces.long()
        self.tangent = tangent
        self.tangent_face = tangent_face
        if tangent_face is not None:
            self.tangent = self.tangent.long()
        if other_mesh is not None:
            self.v_pos = other_mesh.v_pos
            self.pos_faces = other_mesh.pos_faces
            self.nrms = other_mesh.nrms
            self.nrm_faces = other_mesh.nrm_faces

            self.uvs = other_mesh.uvs
            self.uv_faces = other_mesh.uv_faces
            self.tangent = other_mesh.tangent
            self.tangent_face = other_mesh.tangent_face
        self.unit_size()

    def aabb(self):
        return torch.min(self.v_pos, dim=0).values, torch.max(self.v_pos, dim=0).values

    def unit_size(self):
        with torch.no_grad():
            vmin, vmax = self.aabb()
            scale = 2 / torch.max(vmax - vmin).item()
            v_pos = self.v_pos - (vmax + vmin) / 2  # Center mesh on origin
            v_pos = v_pos * scale  # Rescale to unit size
            self.v_pos = v_pos
            # return Mesh(  v_pos, base=mesh)

    def compute_tangents(self):

        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0, 3):
            pos[i] = self.v_pos[self.pos_faces[:, i]]
            tex[i] = self.uvs[self.uv_faces[:, i]]
            vn_idx[i] = self.nrm_faces[:, i]

        tangents = torch.zeros_like(self.nrms)
        tansum = torch.zeros_like(self.nrms)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1 = pos[1] - pos[0]
        pe2 = pos[2] - pos[0]

        nom = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
        denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])

        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0, 3):
            idx = vn_idx[i][:, None].repeat(1, 3)
            tangents.scatter_add_(0, idx, tang)  # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang))  # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = Render_Util.safe_normalize(tangents)
        tangents = Render_Util.safe_normalize(tangents - Render_Util.dot(tangents, self.nrms) * self.nrms)

        self.tangent = tangents
        self.tangent_face = self.nrm_faces

        self.pos_faces = self.pos_faces.int()
        self.nrm_faces = self.nrm_faces.int()
        self.uv_faces = self.uv_faces.int()
        self.tangent_face = self.tangent_face.int()

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

 

 