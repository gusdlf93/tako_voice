import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Encoder(nn.Module):
    def __init__(self, channel=[1, 64, 128, 256], kernel_size=5, stage = 3):
        super(Encoder, self).__init__()

        convolutions = []
        for i in range(stage):
            conv_layer = nn.Sequential(
                ConvNorm(channel[i], channel[i+1], kernel_size=kernel_size, stride=2, padding = 2, dilation=1)
                # nn.BatchNorm1d(channel[i+1])
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        for conv in self.convolutions:
            x = conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channel=[256, 128, 64, 1], kernel_size=5, stage = 3):
        super(Decoder, self).__init__()

        convolutions = []
        for i in range(stage):
            # conv_layer = nn.Sequential(
            #     torch.nn.Upsample(scale_factor=2, mode='linear'),
            #     ConvNorm(channel[i], channel[i+1], kernel_size=kernel_size, stride=1, padding = 2, dilation=1)
            #     # nn.BatchNorm1d(channel[i+1])
            # )
            conv_layer = nn.Sequential(
                #torch.nn.Upsample(scale_factor=2, mode='linear'),
                nn.ConvTranspose1d(channel[i], channel[i+1], kernel_size=kernel_size, stride=2, padding = 2, dilation=1)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x, original_size):
        for conv in self.convolutions:
            x = conv(x)
        if x.shape[2] > original_size: #cut
            x = x[:,:,:original_size]
        elif x.shape[2] < original_size: #padd
            x_padded = torch.zeros(x.shape[0], x.shape[1], original_size)
            # x_padded = torch.FloatTensor(x.shape[0], x.shape[1], original_size)
            x_padded[:, :, :x.shape[2]] = x
            x = x_padded.to('cuda')
        return x

class noise_remover(nn.Module):
    def __init__(self, en_channel, de_channel, kernel_size = 5, stage = 3):
        super().__init__()

        self.Encoder = Encoder(channel=en_channel, kernel_size = 5, stage = stage)
        self.Decoder = Decoder(channel=de_channel, kernel_size = 5, stage = stage)

    def forward(self, x):
        original_size = x.shape[2]
        x = self.Encoder(x)
        x = self.Decoder(x, original_size)
        return x