import torch.nn as nn
import torch
import numpy as np
import pandas as pd
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


def data_Loader(path):
    data = pd.read_csv(path)
    data = np.array(data)
    time_start = time.time()
    y = data[:,5]
    #x = data[:,8:148]
    x = data[:,7:147]
    x = x.astype(float)
    y = y.astype(float)
    x = x[:,np.newaxis,:]
    x = torch.tensor(x).to(torch.float32)
    y = torch.tensor(y).to(torch.float32)
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset=dataset, batch_size=50,shuffle=True)
    print('loading_time：{}'.format(time.time()-time_start))
    return dataloader


class Accumulator:

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):

    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    metric2 = Accumulator(4)
    count1, count2, count3, count4 = 0, 0, 0, 0
    for X, y in data_iter:

        X = X.to(device)
        y = y.to(device, dtype=torch.long)
        y_hat = net(X)
        y_hat = y_hat.argmax(1)
        for i in range(len(y)):
            if (y[i] == 0):
                count1 = count1 + 1
                if (y_hat[i] == 0):
                    count2 = count2 + 1
        for i in range(len(y_hat)):
            if (y_hat[i] == 0):
                count3 = count3 + 1

                if (y[i] == 0):
                    count4 = count4 + 1
        if (count3 == 0):
            count3 = 1
        metric.add(accuracy(net(X), y), y.numel())
        metric2.add(count1, count2, count3, count4)

    return metric[0] / metric[1], metric2[1] / metric2[0], metric2[3] / metric2[2]

def train_epoch (net, train_iter, loss, updater):

    if isinstance(net, torch.nn.Module):
        net.train()
    metric= Accumulator(3)
    for X,y in train_iter:
        X,y = X.to(device),y.to(device,dtype=torch.long)
        y_hat= net(X)
        l= loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):

            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:

            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]


class Animator:  #@save

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):

        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)




def train_ch(net, train_iter, test_iter, loss, num_epochs, updater, patience):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train acc', 'd precision', 'd recall', 'test acc'])
    best_test = 0

    def init_weight(m):
        if type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            # Kaiming or Xavier
            if hasattr(m, 'next_activation') and m.next_activation == 'ReLU':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            else:
                nn.init.xavier_normal_(m.weight)
        elif type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif type(m) == nn.BatchNorm1d:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif type(m) == nn.LSTM:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    net.apply(init_weight)
    scheduler = lr_scheduler.CosineAnnealingLR(updater, T_max=num_epochs)  # 创建余弦退火学习率调度器
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc, pre, rec = evaluate_accuracy(net, test_iter)
        a, b = train_metrics
        print('epoch：%d train_loss:%f train_acc:%f test_acc:%f defect_pre：%f defect_rec：%f'% (
        epoch, a, b, test_acc, pre, rec))
        animator.add(epoch + 1, (b,) + (pre,) + (rec,) + (test_acc,))

        val_test = test_acc

        if val_test > best_test:
            best_test = val_test
            counter = 0
            torch.save(net.state_dict(), 'best_model.pkl')
            best_epoch = epoch
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                net.load_state_dict(torch.load("best_model.pkl"))
                break
        scheduler.step()

    train_acc, b, c = evaluate_accuracy(net, train_iter)
    test_acc, pre, rec = evaluate_accuracy(net, test_iter)

    #     #train_loss,train_acc= train_metrics
    print(
        'best_epoch：%d train_acc:%f test_acc:%f defect_pre：%f defect_rec：%f' % (best_epoch, train_acc, test_acc, pre, rec))


class CIFM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CIFM, self).__init__()
        self.scale_param = nn.Parameter(torch.ones(1))
        self.Linear = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = nn.ReLU()

    def forward(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_param / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        R = self.activation(torch.bmm(attn_weights, V))

        C = self.activation(self.Linear(V - R))

        output = self.a * R + self.b * C

        return output


class Cross_attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Cross_attention, self).__init__()
        self.scale_param = nn.Parameter(torch.ones(1))
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, Q, K, V):

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_param / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.bmm(attn_weights, V)

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
        return output1, output2


class Bi_Cross(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Bi_Cross, self).__init__()
        self.cross = Cross_attention(in_dim=in_dim, out_dim=out_dim)
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

        output1 = self.cross(Q_y, K_x, V_x)
        output2 = self.cross(Q_x, K_y, V_y)
        # output= torch.cat([output1,output2 ],dim=1)
        return output1, output2



warnings.filterwarnings('ignore')
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

        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # return x*self.sigmoid(out)

        #  (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,8,8) -> (2,512,1,1)
        avg = self.avg_pool(x)
        # mlp (2,512,8,8) -> (2,512,1,1) -> (2,512/ration,1,1) -> (2,512,1,1)
        # (2,512,1,1) -> (2,512/ratio,1,1)
        avg = self.fc1(avg)
        avg = self.relu1(avg)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        avg_out = self.fc2(avg)


        # (2,512,8,8) -> (2,512,1,1)
        max = self.max_pool(x)

        # (2,512,1,1) -> (2,512/ratio,1,1)
        max = self.fc1(max)
        max = self.relu1(max)
        # (2,512/ratio,1,1) -> (2,512,1,1)
        max_out = self.fc2(max)

        # (2,512,1,1) + (2,512,1,1) -> (2,512,1,1)
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
        # print('x shape', x.shape)
        # (2,512,8,8) -> (2,1,8,8)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # (2,512,8,8) -> (2,1,8,8)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # (2,1,8,8) + (2,1,8,8) -> (2,2,8,8)
        cat = torch.cat([avg_out, max_out], dim=1)
        # (2,2,8,8) -> (2,1,8,8)
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

if __name__ == "__main__":
    path_train = '../Data/train.csv'
    path_test = '../Data/test.csv'

    trainLoader = data_Loader(path_train)
    testLoader = data_Loader(path_test)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = DFFC()

    model.to(device)
    num_epochs = 100
    loss = nn.CrossEntropyLoss()
    lr = 0.1
    patience = 30
    trainer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    print('train on :', device)
    train_ch(model, trainLoader, testLoader, loss, num_epochs, trainer, patience)