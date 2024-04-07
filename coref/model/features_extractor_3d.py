import torch
import torch.nn as nn
import torch.nn.functional as F


class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()

        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)

    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)


class Vox3DCNN(nn.Module):
    def __init__(self, features_dim, dropout=0.0, first_stride=2, out_len=64):

        super(Vox3DCNN, self).__init__()

        # Encoder architecture
        stride_tuple = (first_stride, first_stride, first_stride)
        self.out_len = out_len  # int((4/first_stride) ** 3)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32,
                               kernel_size=4, stride=stride_tuple, padding=(2,
                                                                            2, 2))
        self.b1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b3 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b4 = nn.BatchNorm3d(num_features=256)

        # this sequential module is created for multi gpu training.
        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.Dropout(dropout),
            self.b4,
        )

        self.ll = DMLP(256, 256, 256, features_dim, dropout)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        out = input.unsqueeze(1)
        out = self._encoder(out)

        out = out.view(-1, 256, self.out_len)
        out = out.transpose(1, 2)
        out = self.ll(out)
        return out
