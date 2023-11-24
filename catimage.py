import imageio.v2 as imageio
import torch

i1 = imageio.imread('./1.png')
i2 = imageio.imread('./2.png')
i3 = imageio.imread('./3.png')

i1 = torch.from_numpy(i1)
i2 = torch.from_numpy(i2)
i3 = torch.from_numpy(i3)

res = torch.cat([i1, i2, i3], -2)
imageio.imwrite("out.png", res)

eee = 0
