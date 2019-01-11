# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class StorableNet(nn.Module):
    def __init__(self):
        super(StorableNet, self).__init__()

    def forward(self, *inputs):
        raise NotImplementedError

    def load(self, path, use_gpu=False):
        if use_gpu:
            self.load_state_dict(
                torch.load(path, map_location=lambda s, l: s.cuda()))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class NaiveNetwork(StorableNet):
    def __init__(self, out_channels):
        super(NaiveNetwork, self).__init__()
        self.out_channels = out_channels
        self.conv41 = nn.Conv2d(
            in_channels=12,  # input height
            out_channels=out_channels,  # n_filter
            kernel_size=(4, 1),  # filter size
        )

        self.conv14 = nn.Conv2d(
            in_channels=12,  # input height
            out_channels=out_channels,  # n_filter
            kernel_size=(1, 4),  # filter size
        )

        self.conv22 = nn.Conv2d(
            in_channels=12,  # input height
            out_channels=out_channels,  # n_filter
            kernel_size=(2, 2),  # filter size
        )

        self.conv33 = nn.Conv2d(
            in_channels=12,  # input height
            out_channels=out_channels,  # n_filter
            kernel_size=(3, 3),  # filter size
        )

        self.conv44 = nn.Conv2d(
            in_channels=12,  # input height
            out_channels=out_channels,  # n_filter
            kernel_size=(4, 4),  # filter size
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels * (4 + 4 + 9 + 4 + 1), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
        )

    def forward(self, state):
        c41 = F.relu(self.conv41(state))
        c14 = F.relu(self.conv14(state))
        c22 = F.relu(self.conv22(state))
        c33 = F.relu(self.conv33(state))
        c44 = F.relu(self.conv44(state))
        cs = [c.view(c.shape[0], -1) for c in [c41, c14, c22, c33, c44]]
        result = torch.cat(cs, 1)
        return self.fc(result)

    def predict(self, onehot_board, use_gpu=False):
        x = torch.Tensor(onehot_board).view(-1, 12, 4, 4)
        if use_gpu:
            x = x.cuda()
        y = self.forward(x)
        return int(torch.argmax(y))
