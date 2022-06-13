from collections import namedtuple
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util
from .base import RNNBase

LSTMState = namedtuple('LSTMState', ('hidden', 'cell'))


class ConvolutionModule(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        self.convs = nn.ModuleList([
            nn.Conv2d(
                config[i - 1].ch if i > 0 else self.input_shape.channel,
                config[i].ch,
                config[i].kernel_size,
                config[i].stride,
                config[i].padding,
            ) for i in range(len(config))
        ])

        for conv in self.convs:
            util.init_conv(conv, 'relu', 'he')

    def get_convolved_shape(self):
        device = next(self.parameters()).device
        x = torch.zeros(1, self.input_shape.channel, self.input_shape.height,
                        self.input_shape.width).to(device)
        for conv in self.convs:
            x = conv(x)

        return list(x.shape[1:])

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))

        return x


class DeconvolutionModule(nn.Module):
    def __init__(self, config, convolved_shape):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.ConvTranspose2d(
                config[i - 1].ch if i > 0 else convolved_shape[0],
                config[i].ch,
                config[i].kernel_size,
                config[i].stride,
                config[i].padding,
            ) for i in range(len(config))
        ])
        for conv in self.convs:
            util.init_conv_transposed(conv, 'relu', 'he')

        util.init_conv_transposed(self.convs[-1], 'tanh', 'xavier')

    def forward(self, x):

        for conv in self.convs[:-1]:
            x = F.relu(conv(x))

        x = torch.tanh(self.convs[-1](x))

        return x


class VisionEncoderModule(nn.Module):
    def __init__(self, config, lns):
        super().__init__()
        self.config = config
        self.conv = ConvolutionModule(config.conv, config.input)

        convolved_shape = self.conv.get_convolved_shape()
        fc_input_size = reduce(lambda x, y: x * y, convolved_shape)

        self.fcs = nn.ModuleList([
            nn.Linear(config.fc[i].input if i > 0 else fc_input_size,
                      config.fc[i].output) for i in range(len(config.fc))
        ])

        self.lns = lns

        for fc in self.fcs:
            util.init_fc(fc, 'relu', 'he')

    def get_convolved_shape(self):
        return self.conv.get_convolved_shape()

    def forward(self, x):
        x = self.conv(x)

        x = x.reshape(x.size(0), -1)

        for fc, ln in zip(self.fcs, self.lns):
            x = F.relu(ln(fc(x)))
        return x


class IntegrationModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(2 * config.fc.each_input, config.fc.output)

        util.init_fc(self.fc, 'relu', 'he')

    def forward(self, sh, oh):
        x = torch.cat([sh, oh], 1)
        x = F.relu(self.fc(x))
        return x


class VisionDecoderModule(nn.Module):
    def __init__(self, config, convolved_shape):
        super().__init__()
        self.config = config

        self.conv = DeconvolutionModule(config.conv, convolved_shape)
        self.convolved_shape = convolved_shape

        fc_output_size = reduce(lambda x, y: x * y, self.convolved_shape)

        self.fcs = nn.ModuleList([
            nn.Linear(
                config.fc[i].input, config.fc[i].output if i <
                (len(config.fc) - 1) else fc_output_size)
            for i in range(len(config.fc))
        ])

        for fc in self.fcs:
            util.init_fc(fc, 'relu', 'he')

    def forward(self, x):
        for fc in self.fcs:
            x = F.relu(fc(x))

        x = x.reshape(*([x.size(0)] + self.convolved_shape))

        x = self.conv(x)
        return x


class SuperpositionModule(nn.Module, RNNBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTMCell(config.input.motion + config.input.vision,
                                config.hidden)

    def init_state(self, batch_size):
        device = next(self.lstm.parameters()).device
        init_sh = torch.zeros((batch_size, self.config.hidden), device=device)
        init_sc = torch.zeros((batch_size, self.config.hidden), device=device)
        init_oh = torch.zeros((batch_size, self.config.hidden), device=device)
        init_oc = torch.zeros((batch_size, self.config.hidden), device=device)
        self.state = {
            'self': LSTMState(init_sh, init_sc),
            'other': LSTMState(init_oh, init_oc),
        }

    def forward(self, sv, sm, ov, om):

        sh, sc = self.state['self']
        oh, oc = self.state['other']

        sx = torch.cat([sv, sm], 1)
        ox = torch.cat([ov, om], 1)

        next_sh, next_sc = self.lstm(sx, (sh, sc))
        next_oh, next_oc = self.lstm(ox, (oh, oc))

        self.state['self'] = LSTMState(next_sh, next_sc)
        self.state['other'] = LSTMState(next_oh, next_oc)

        return next_sh, next_oh


class MotionGeneratorModule(nn.Module, RNNBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTMCell(config.input, config.hidden)
        self.fc = nn.Linear(config.hidden, config.output)

        util.init_fc(self.fc, 'tanh', 'xavier')

    def init_state(self, batch_size):
        device = next(self.lstm.parameters()).device
        init_h = torch.zeros((batch_size, self.config.hidden), device=device)
        init_c = torch.zeros((batch_size, self.config.hidden), device=device)
        self.state = {
            'motion_generator': LSTMState(init_h, init_c),
        }

    def forward(self, x):

        h, c = self.state['motion_generator']

        next_h, next_c = self.lstm(x, (h, c))

        self.state['motion_generator'] = LSTMState(next_h, next_c)

        o = torch.tanh(self.fc(next_h))

        return o


class FeaturePredictionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fcs = nn.ModuleList([
            nn.Linear(config.fc[i].input, config.fc[i].output)
            for i in range(len(config.fc))
        ])

        self.lns = nn.ModuleList(
            [nn.LayerNorm(config.fc[i].output) for i in range(len(config.fc))])

        for fc in self.fcs:
            util.init_fc(fc, 'relu', 'he')

    def forward(self, x):

        for fc, ln in zip(self.fcs, self.lns):
            x = F.relu(ln(fc(x)))
        return x
