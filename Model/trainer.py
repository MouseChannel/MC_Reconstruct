import torch

from RenderUtil import RenderContext


class Trainer(torch.nn.Module):
    def __init__(self, trainable_go):
        super(Trainer, self).__init__()
        self.trainable_go = trainable_go

    def forward(self, mvp, campos, light_pos, light_power):
        inference_image = RenderContext.render_context.rasterize(self.trainable_go,
                                                                 mvp,
                                                                 1024,
                                                                 1024,
                                                                 campos,
                                                                 light_pos,
                                                                 light_power)

        return inference_image
