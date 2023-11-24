import argparse
import json

import imageio
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from Material import Texture2D, Material
from Model.trainer import Trainer
from Object import GameObject
from RenderUtil import obj_loader, Render_Util, RenderContext
from RenderUtil.CameraHelper import get_random_camera_mvp, get_random_camera_pos


def main():
    parser = argparse.ArgumentParser(description='diffmodeling')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config) as f:
            data = json.load(f)
            for key in data:
                print(key, data[key])
                args.__dict__[key] = data[key]

    trainable_diffuse_texture = Texture2D.Texture2D(torch.rand(1024, 1024, 3, device='cuda', requires_grad=True))
    trainable_arm_texture = Texture2D.Texture2D(torch.rand(1024, 1024, 3, device='cuda', requires_grad=True))
    trainable_nrm_texture = Texture2D.Texture2D(torch.rand(1024, 1024, 3, device='cuda', requires_grad=True))
    trainable_mesh = obj_loader.load_obj_mesh(args.base_mesh)
    trainable_mesh.to_be_trainable()
    trainable_material = Material.Material(trainable_diffuse_texture, trainable_arm_texture, trainable_nrm_texture)
    trainable_go = GameObject.Gameobject(trainable_mesh, trainable_material)

    ref_go = obj_loader.load_obj(args.ref_mesh)
    ###

    eye = np.array([.5, 1.0, -.5])
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
    optimizer = torch.optim.Adam([trainable_mesh.v_pos,
                                  trainable_material.diffuse_texture.data,
                                  trainable_material.arm.data,
                                  trainable_material.nrm.data

                                  ], lr=1e-3)
    writer = SummaryWriter('./output/tensorboard')
    trainer = Trainer(trainable_go)
    for current_iter in trange(args.iter):
        ###########random camera_pos###############
        random_eyepos = get_random_camera_pos(1.5)

        mvp = get_random_camera_mvp(1.5, random_eyepos)
        mvp = torch.from_numpy(mvp).cuda()
        campos = torch.from_numpy(random_eyepos).cuda()

        with torch.no_grad():
            ref_image = RenderContext.render_context.rasterize(ref_go, mvp, 1024, 1024, campos, light_pos,
                                                               light_power)
       
        inference_image = trainer(mvp, campos, light_pos, light_power)
        loss = torch.nn.functional.mse_loss(ref_image, inference_image)
        writer.add_scalar("tag", loss, global_step=None, walltime=None)
        # writer.add_graph(trainer)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        ########################################################
        ######################  display  #######################
        ########################################################
        if current_iter % 100 == 0:
            result_ref_image = ref_image.squeeze(0).cpu().detach()
            result_inference_image = inference_image.squeeze(0).cpu().detach()
            result_image = torch.cat([result_ref_image, result_inference_image], -2)
            imageio.v2.imsave("output/test" + str(current_iter) + ".png",
                              np.clip(np.rint(result_image.numpy() * 255.0), 0, 255).astype(np.uint8))

        # if current_iter % 120 == 0:
        #     result_ref_image = ref_image[0].squeeze(0).cpu().detach()
        #     result_inference_image = ref_image[1].squeeze(0).cpu().detach()
        #     result_inference_image1 = ref_image[2].squeeze(0).cpu().detach()
        #     result_image = torch.cat([result_ref_image, result_inference_image], -2)
        #     result_image = torch.cat([result_image, result_inference_image1], -2)

            imageio.v2.imsave("output/test" + str(current_iter) + ".png",
                              np.clip(np.rint(result_image.numpy() * 255.0), 0, 255).astype(np.uint8))


if __name__ == '__main__':
    main()
