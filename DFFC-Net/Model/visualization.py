import os
import time
import json
import numpy as np
import cv2
import torch
import pandas as pd
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import init
from torch import Tensor
import math
import warnings
import time
from torch.utils.data import Dataset, TensorDataset, DataLoader,random_split
from d2l import torch as d2l
from IPython import display
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.optim import AdamW
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
label = ['defect', 'badpaint', 'heatReflection', 'post', 'noise']

class CIFM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CIFM, self).__init__()
        self.scale_param = nn.Parameter(torch.ones(1))
        self.Linear = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.activation = nn.Sigmoid()
        self.activation = nn.ReLU()

    def forward(self, Q, K, V):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_param / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        R = self.activation(torch.bmm(attn_weights, V))

        C = self.activation(self.Linear(V - R))

        output = self.a * R + self.b * C

        return output



class BFFM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFFM, self).__init__()
        self.CIFM1 = CIFM(in_dim=in_dim, out_dim=out_dim)
        self.CIFM2 = CIFM(in_dim=in_dim, out_dim=out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim, bias=False)
        self.key = nn.Linear(in_dim, out_dim, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_queries = x.shape[1]
        num_keys = y.shape[1]


        V_x = self.value(x)
        V_y = self.value(y)

        Q_y = self.query(y)
        K_x = self.key(x)

        Q_x = self.query(x)
        K_y = self.key(y)

        output1 = self.CIFM1(Q_y, K_x, V_x) + x
        output2 = self.CIFM1(Q_x, K_y, V_y) + y
        # output2 = self.CIFM2(Q_x, K_y, V_y) + y
        # output= torch.cat([output1,output2 ],dim=1)
        return output1, output2


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=(in_channels//2), kernel_size=1),
            nn.BatchNorm1d(num_features=(in_channels//2)),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=(in_channels//2), out_channels=(in_channels//2), kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=(in_channels//2)),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=(in_channels//2), out_channels=out_channels, kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.dropout(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio):

        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.max_pool = nn.AdaptiveMaxPool1d(1)


        self.fc1 = nn.Conv1d(in_channel, in_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channel // ratio, in_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):


        avg = self.avg_pool(x)

        avg = self.fc1(avg)
        avg = self.relu1(avg)
        avg_out = self.fc2(avg)


        max = self.max_pool(x)

        max = self.fc1(max)
        max = self.relu1(max)

        max_out = self.fc2(max)

        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):

        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat)
        return x * self.sigmoid(out)


class CBAM_block(nn.Module):
    def __init__(self, in_channels, ratio, kernel_size=7):
        super(CBAM_block, self).__init__()
        self.ca = ChannelAttention(in_channel=in_channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)

        return out


class SEAttention1d(nn.Module):
    '''
    Modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    '''
    def __init__(self, channel, reduction):
        super(SEAttention1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b,c,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1)

        return x*y.expand_as(x)


class DFFC(nn.Module):
    def __init__(self, in_channels=2, out_channels=128, ratio=16, dropout=0.2, num_classes=5, kernel_size_s=7):
        super(DFFC, self).__init__()

        self.cross = BFFM(in_dim=70, out_dim=70)
        self.cross1 = BFFM(in_dim=70, out_dim=70)
        self.cross2 = BFFM(in_dim=70, out_dim=70)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2,
                      kernel_size=3, padding=1),
            nn.Conv1d(in_channels=out_channels * 2, out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2,
                      kernel_size=3, padding=1),
            nn.Conv1d(in_channels=out_channels * 2, out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )
        self.bott1 = Bottleneck(in_channels=(out_channels * 2), out_channels=out_channels, dropout=dropout)

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Conv1d(in_channels=out_channels, out_channels=(out_channels * 2),
                      kernel_size=5, padding=2),
            nn.Conv1d(in_channels=(out_channels * 2), out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

        self.s = SpatialAttention(kernel_size=kernel_size_s)
        self.c = ChannelAttention(in_channel=out_channels, ratio=ratio)

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),

        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            # nn.Dropout(dropout),
        )
        self.bott2 = Bottleneck(in_channels=(out_channels * 2), out_channels=out_channels, dropout=dropout)
        self.bott3 = Bottleneck(in_channels=(out_channels * 2), out_channels=out_channels, dropout=dropout)
        # 修改
        self.se1 = SEAttention1d(channel=out_channels, reduction=ratio)
        self.se2 = SEAttention1d(channel=out_channels, reduction=ratio)

        self.bott4 = Bottleneck(in_channels=(out_channels * 2), out_channels=out_channels, dropout=dropout)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.Linear(64, 32),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        co1, co2 = self.cross(x[:, :, 0:70], x[:, :, 70:140])

        x = torch.cat([co1, co2], dim=1)

        x1 = self.conv1(x)

        x2 = self.conv2(x1)

        x2_2 = self.se1(x2)

        x3 = self.conv3(x2_2)
        x4 = self.conv4(x2_2)

        x_s = self.s(x3)
        x_c = self.c(x4)

        x_s2 = self.conv6(x_s)
        # x8 = self.conv7(x6)
        x_c2 = self.conv8(x_c)

        x5_1, x5_2 = self.cross2(x_s2, x_c2)

        x5 = torch.cat([x5_1, x5_2], dim=1)

        x6 = self.bott2(x5)

        out = self.avg_pool(x6)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
# image_color_fill
def fill_image_region(image, matrix, matrix2, color_mapping):
    for i in range(len(matrix)):
        print(matrix2[i])
        dominant_class = matrix2[i].argmax(-1)
        if (dominant_class == 0 and matrix2[i][0] == 0):
            color = [255, 0, 255]
        else:
            color = color_mapping[dominant_class.item()]
        print(color)
        x_min, x_max, y_min, y_max = matrix[i].int()
        #         for i in range(y_min,y_max):
        #             for j in range(x_min,x_max):
        #                 if(image[i][j]==255):
        #                     image[i][j]=color
        region = image[y_min:y_max, x_min:x_max]
        mask = (region > [200, 200, 200]).all(axis=2)
        # mask = (region != [0, 0, 0]).all(axis = 2)
        region[mask] = color
    return image


def detect_img(path, net, save_name, label_path):
    t0 = time.time()
    dirs = os.listdir(path)
    color_mapping = {  # BGR格式
        0: [255, 0, 255],  # defect
        1: [0, 255, 0],  # badpaint
        2: [0, 0, 255],  # heatReflection
        3: [0, 255, 255]  # post
    }

    for dir in dirs:
        a = 0
        name = dir[:-4]
        print("%sloading......." % name)
        path1 = os.path.join(path, dir)
        read_img = os.path.join(label_path, name + '.jpg')
        json_path = os.path.join(label_path, name + '.json')
        save_img = os.path.join(save_name, name + '.jpg')

        # 文件存在性检查
        if not os.path.exists(path1):
            print(f"file {path1} does not exist，skip")
            continue
        if not os.path.exists(read_img):
            print(f"file {read_img} does not exist，skip")
            continue
        if not os.path.exists(json_path):
            print(f"file {json_path} does not exist，skip")
            continue

        try:
            data = pd.read_csv(path1)
        except Exception as e:
            print(f"load {path1} error: {e}，skip")
            continue
        data = np.array(data)
        try:
            image = cv2.imread(read_img)
        except Exception as e:
            print(f"load{read_img} error: {e}，skip")
            continue
        try:
            with open(json_path, 'r') as f:
                data_json = json.load(f)
        except Exception as e:
            print(f"load {json_path} error: {e}，skip")
            continue

        shapes = data_json["shapes"]
        L = len(shapes)
        matrix = torch.zeros(L, 4)
        matrix2 = torch.zeros(L, 4)
        for i in range(len(shapes)):
            points = shapes[i]['points']
            x_min, y_min = np.min(points, axis=0).astype(int)
            x_max, y_max = np.max(points, axis=0).astype(int)
            matrix[i][0] = x_min
            matrix[i][1] = x_max
            matrix[i][2] = y_min
            matrix[i][3] = y_max
        for i in range(len(data)):
            name_i = data[i][0]
            label_i = data[i][1]
            x = data[i][2]
            y = data[i][3]
            test = data[i][7:147]
            test = torch.tensor(test.astype(float)).to(torch.float32)
            test = test.reshape(1, 1, -1)
            test = test.to(device)
            pre = net(test).argmax(1)

            for i in range(L):
                if (matrix[i][0] <= x and x <= matrix[i][1]) and (matrix[i][2] <= y and y <= matrix[i][3]):
                    if pre == torch.tensor(0):
                        a += 1
                        matrix2[i][0] += 1
                    if pre == torch.tensor(1):
                        matrix2[i][1] += 1
                    if pre == torch.tensor(2):
                        matrix2[i][2] += 1
                    if pre == torch.tensor(3):
                        matrix2[i][3] += 1
        print(a)
        image = fill_image_region(image, matrix, matrix2, color_mapping)
        cv2.imwrite(save_img, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    t = time.time() - t0
    print('detecting time:%d' % t)

if __name__ == "__main__":
    net = DFFC()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load('model.pkl', map_location=device))  # loading the best_model
    net.to(device)
    net.eval()
    label_path = "../path/labelme/data_test"
    data_path = '../Data/data_test'
    save_path = "../Results"
    detect_img(data_path, net, save_path, label_path)

