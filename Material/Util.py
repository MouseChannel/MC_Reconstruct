import os
from . import Texture2D


# import texture2D


def load_mtl(fn):
    import re
    mtl_path = os.path.dirname(fn)

    # Read file
    with open(fn) as f:
        lines = f.readlines()

    # Parse materials
    materials = []
    for line in lines:
        split_line = re.split(' +|\t+|\n+', line.strip())
        prefix = split_line[0].lower()
        data = split_line[1:]
        if 'newmtl' in prefix:
            material = {'name': data[0]}
            materials += [material]
        elif materials:
            if 'bsdf' in prefix or 'map_kd' in prefix or 'map_ks' in prefix or 'map_bump' in prefix:
                material[prefix] = data[0]
            # else:
            #     material[prefix] = torch.tensor(tuple(float(d) for d in data), dtype=torch.float32, device='cuda')

    # Convert everything to textures. Our code expects 'kd' and 'ks' to be texture maps. So replace constants with 1x1 maps
    for mat in materials:
        # if not 'bsdf' in mat:
        #     mat['bsdf'] = 'pbr'

        # mat['kd'] = texture2D.load_texture2D(os.path.join(mtl_path, mat['map_kd']))
        mat['kd'] = Texture2D.load_texture2D(mat['map_kd'])

        # mat['ks'] = texture2D.load_texture2D(os.path.join(mtl_path, mat['map_ks']), channels=3)
        mat['ks'] = Texture2D.load_texture2D(mat['map_ks'], channels=3)

        if 'map_bump' in mat:
            # mat['normal'] = texture2D.load_texture2D(os.path.join(mtl_path, mat['map_bump']),
            #                                          lambda_fn=lambda x: x * 2 - 1,
            #                                          channels=3)
            mat['normal'] = Texture2D.load_texture2D(mat['map_bump'], lambda_fn=lambda x: x * 2 - 1,
                                                     channels=3)

        # Convert Kd from sRGB to linear RGB
        mat['kd'].data = Texture2D.srgb_to_rgb(mat['kd'].data)

        # if clear_ks:
        #     # Override ORM occlusion (red) channel by zeros. We hijack this channel
        #     for mip in mat['ks'].getMips():
        #         mip[..., 0] = 0.0

    return materials
# raise RuntimeError('no mtllib_file in obj')
