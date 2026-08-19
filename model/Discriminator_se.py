#PyTorch lib
import torch.nn as nn
import torch as torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
import sys
from datetime import datetime
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)



class Concatenate(nn.Module):
    def __init__(self, dim=-1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class CompletionNetwork(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        # input_shape: (None, 4, img_h, img_w)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()
        # input_shape: (None, 256, img_h//4, img_w//4)
        self.deconv13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()
        # input_shape: (None, 128, img_h//2, img_w//2)
        self.deconv15 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()
        # input_shape: (None, 64, img_h, img_w)
        self.conv16 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()
        # input_shape: (None, 32, img_h, img_w)
        self.conv17 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.act17 = nn.Sigmoid()
        # output_shape: (None, 3, img_h. img_w)

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))
        return x


class ContextDiscriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape, arc='celeba'):
        super(ContextDiscriminator, self).__init__()
        self.arc = arc
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape, arc=arc)
        # input_shape: [(None, 1024), (None, 1024)]
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        self.concat1 = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        self.linear1 = nn.Linear(in_features, 1)
        self.act1 = nn.Sigmoid()
        # output_shape: (None, 1)

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        out = self.act1(self.linear1(self.concat1([x_ld, x_gd])))
        return out
#Model
class Discriminator_1(nn.Module):
    def __init__(self):
        super(Discriminator_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.ReLU()
            )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(128, 1, 5, 1, 2)
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 64, 5, 4, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 32, 5, 4, 1),
            nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(10560, 1024),
            # nn.Linear(32, 1024),
            nn.Linear(1024, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        # x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return mask, x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv0 = nn.Conv2d(62, 64, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        self.conv9 = nn.Conv2d(512, 1, 1, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        relu = self.relu
        N, C, H, W = x.shape
        if C == 3:
            x = relu(self.conv1(x))
        else:
            x = relu(self.conv0(x))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        x = relu(self.bn5(self.conv5(x)))
        x = relu(self.bn6(self.conv6(x)))
        x = relu(self.bn7(self.conv7(x)))
        x = relu(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return self.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)

def calculate_loss_D(input_high, out, seg_map):

    N, C, H, W = input_high.shape
    max_1_seg = torch.zeros(N, 1, H, W).cuda().clone()
    max_2_seg = torch.zeros(N, 1, H, W).cuda().clone()
    seg_cls = seg_map.reshape(N, seg_map.shape[1], -1)
    # seg_cls = seg_cls.argmax(dim=1)
    seg_cls_1 = torch.sum(seg_cls, dim=2)  # N x cls
    cls_max_1 = seg_cls_1.argmax(dim=1)   # N
    for i in range(N):
        seg_cls_1[i, cls_max_1[i]] = 0
    cls_max_2 = seg_cls_1.argmax(dim=1)
    for i in range(N):
        max_1_seg[i, :, :, :] = seg_map[i, cls_max_1[i], :, :]
        max_2_seg[i, :, :, :] = seg_map[i, cls_max_2[i], :, :]


    input_high_seg_1 = input_high * torch.cat((max_1_seg, max_1_seg, max_1_seg), dim=1)
    out_seg_1 = out * torch.cat((max_1_seg, max_1_seg, max_1_seg), dim=1)


    return input_high_seg_1, out_seg_1

