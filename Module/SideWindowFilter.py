#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-23 16:32
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : SideWindowFilter.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SideWindowFilter(nn.Module):

    def __init__(self, radius, iteration):
        super(SideWindowFilter, self).__init__()
        self.radius = radius
        self.iteration = iteration
        self.kernel_size = 2 * self.radius + 1
        self.filter = nn.Parameter(torch.Tensor(1, 1, self.kernel_size, self.kernel_size))

    def forward(self, im):
        b, c, h, w = im.size()

        d = torch.zeros(b, 8, h, w, dtype=torch.float)
        res = im.clone()

        L, R, U, D = [self.filter.clone() for _ in range(4)]

        L[:, :, :, self.radius + 1:] = 0
        R[:, :, :, 0: self.radius] = 0
        U[:, :, self.radius + 1:, :] = 0
        D[:, :, 0: self.radius, :] = 0

        NW, NE, SW, SE = U.clone(), U.clone(), D.clone(), D.clone()

        L, R, U, D = L / ((self.radius + 1) * self.kernel_size), R / ((self.radius + 1) * self.kernel_size), \
                     U / ((self.radius + 1) * self.kernel_size), D / ((self.radius + 1) * self.kernel_size)

        NW[:, :, :, self.radius + 1:] = 0
        NE[:, :, :, 0: self.radius] = 0
        SW[:, :, :, self.radius + 1:] = 0
        SE[:, :, :, 0: self.radius] = 0

        NW, NE, SW, SE = NW / ((self.radius + 1) ** 2), NE / ((self.radius + 1) ** 2), \
                         SW / ((self.radius + 1) ** 2), SE / ((self.radius + 1) ** 2)

        print('L:', L)
        print('R:', R)
        print('U:', U)
        print('D:', D)
        print('NW:', NW)
        print('NE:', NE)
        print('SW:', SW)
        print('SE:', SE)

        for ch in range(c):
            im_ch = im[:, ch, ::].clone().view(b, 1, h, w)
            # print('im size in each channel:', im_ch.size())

            for i in range(self.iteration):
                # print('###', (F.conv2d(input=im_ch, weight=L, padding=(self.radius, self.radius)) / sum_L -
                # im_ch).size(), d[:, 0,::].size())
                d[:, 0, ::] = F.conv2d(input=im_ch, weight=L, padding=(self.radius, self.radius)) - im_ch
                d[:, 1, ::] = F.conv2d(input=im_ch, weight=R, padding=(self.radius, self.radius)) - im_ch
                d[:, 2, ::] = F.conv2d(input=im_ch, weight=U, padding=(self.radius, self.radius)) - im_ch
                d[:, 3, ::] = F.conv2d(input=im_ch, weight=D, padding=(self.radius, self.radius)) - im_ch
                d[:, 4, ::] = F.conv2d(input=im_ch, weight=NW, padding=(self.radius, self.radius)) - im_ch
                d[:, 5, ::] = F.conv2d(input=im_ch, weight=NE, padding=(self.radius, self.radius)) - im_ch
                d[:, 6, ::] = F.conv2d(input=im_ch, weight=SW, padding=(self.radius, self.radius)) - im_ch
                d[:, 7, ::] = F.conv2d(input=im_ch, weight=SE, padding=(self.radius, self.radius)) - im_ch

                d_abs = torch.abs(d)
                print('im_ch', im_ch)
                print('dm = ', d_abs.shape, d_abs)
                mask_min = torch.argmin(d_abs, dim=1, keepdim=True)
                print('mask min = ', mask_min.shape, mask_min)
                dm = torch.gather(input=d, dim=1, index=mask_min)
                im_ch = dm + im_ch

            res[:, ch, ::] = im_ch
        return res


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.filter = SideWindowFilter(radius=1, iteration=1)

    def forward(self, x):
        return self.filter(x)


if __name__ == '__main__':
    s = SideWindowFilter(radius=1, iteration=1)
    from PIL import Image

    import cv2
    img = cv2.imread('../lena.png', flags=0)
    img = torch.tensor(img, dtype=torch.float)
    print('img ori = ', img)

    print(len(img.size()))
    if len(img.size()) == 2:
        h, w = img.size()
        img = img.view(-1, 1, h, w)
    else:
        c, h, w = img.size()
        img = img.view(-1, c, h, w)
    print('img = ', img.shape)

    model = Net()
    res = model(img)
    res.mean().backward()

    print('res = ', res.shape, res)
    import numpy as np
    if res.size(1) == 3:
        img_res = np.transpose(np.squeeze(res.data.numpy()), (1, 2, 0))
    else:
        img_res = np.squeeze(res.data.numpy())

    # print(img_res.shape, img_res)
    img_res = img_res
    img_res = img_res.astype(np.uint8)
    print('img res:', img_res)
    img_res = Image.fromarray(img_res)  # numpy to image
    img_res.show()