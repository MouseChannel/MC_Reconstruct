import imageio.v2
import torch

from RenderUtil import obj_loader, Render_Util, RenderContext
import numpy as np

###

eye = np.array([1., 1.0, -1.])
up = np.array([0.0, 1.0, 0.0])
at = np.array([0, 0, 0])
a_mv = Render_Util.lookAt(eye, at, up)
proj = Render_Util.projection()
a_mvp = np.matmul(proj, a_mv).astype(np.float32)[None, ...]
a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
a_campos = np.linalg.inv(a_mv)[None, :3, 3]
light_power = torch.tensor(5., device='cuda')

mvp = torch.from_numpy(a_mvp).cuda()
campos = torch.from_numpy(a_campos).cuda()
light_pos = torch.from_numpy(a_lightpos).cuda()
###

go = obj_loader.load_obj("/mocheng/MC_Reconstruct/data/sphere.obj")

canvas = RenderContext.render_context.rasterize(go, mvp, 1024, 1024, campos, light_pos, light_power)
result_image = canvas.squeeze(0).cpu().detach().numpy()
imageio.v2.imsave("test.png", np.clip(np.rint(result_image * 255.0), 0, 255).astype(np.uint8))
aaa = 0
