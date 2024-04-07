import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .features_extractor_3d import DMLP


class Vox2DCNN(nn.Module):
    def __init__(self, features_dim, dropout=0.0, first_stride=2, out_len=16):

        super(Vox2DCNN, self).__init__()

        in_channels = 1  # observation_space.shape[0]
        self.output_dim = features_dim
        self.out_len = out_len  # int((4/first_stride) ** 3)
        modules = []
        modules.extend(self.get_conv_pack(in_channels, 8 * 2, dropout))
        modules.extend(self.get_conv_pack(8*2, 16 * 2, dropout))
        modules.extend(self.get_conv_pack(16*2, 32 * 2, dropout))
        modules.extend(self.get_conv_pack(32 * 2, 64 * 4, dropout))
        # modules.extend([nn.Flatten()])
        self.extractor = nn.Sequential(*modules)
        self.ll = DMLP(256, 256, 256, features_dim, dropout)

        # self.initialize_weights()
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def get_conv_pack(self, in_channels, out_channels, dropout):
        output = [
            add_coord(),
            nn.Conv2d(in_channels + 2, out_channels, 3, padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2)
        ]
        return output

    def forward(self, x_in):
        # (X, 64, 64) INPUT
        x_in = x_in.unsqueeze(1)
        out = self.extractor(x_in)
        out = out.view(-1, 256, self.out_len)
        out = out.transpose(1, 2)
        out = self.ll(out)
        return out


class add_coord(nn.Module):
    def __init__(self):
        super(add_coord, self).__init__()
        self.bs = None
        self.ch = None
        self.h_coord = None
        self.w_coord = None

    def forward(self, x):
        if self.bs == None:
            bs, ch, h, w = x.size()
            self.h_coord = th.arange(
                start=0, end=h).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat([bs, 1, 1, w])/(h/2)-1
            self.w_coord = th.arange(
                start=0, end=w).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat([bs, 1, h, 1])/(w/2)-1
            self.h_coord = self.h_coord.cuda()
            self.w_coord = self.w_coord.cuda()
        h_coord = self.h_coord  # .clone()
        w_coord = self.w_coord  # .clone()
        return th.cat([x, h_coord, w_coord], dim=1)
