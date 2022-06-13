import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util
from .base import WithRNNModuleBase
from .modules import (FeaturePredictionModule, IntegrationModule, MotionGeneratorModule,
                      SuperpositionModule, VisionDecoderModule,
                      VisionEncoderModule)


def add_vision_encoder_module(self, config):

    self.share_lns = nn.ModuleList([
        nn.LayerNorm(config.vision_encoder_module.fc[i].output)
        for i in range(len(config.vision_encoder_module.fc))
    ])
    self.self_vision_encoder_module = VisionEncoderModule(
        config.vision_encoder_module, self.share_lns)
    self.other_vision_encoder_module = VisionEncoderModule(
        config.vision_encoder_module, self.share_lns)


def add_integration_module(self, config):
    self.integration_module = IntegrationModule(config.integration_module)


def add_feature_prediction_module(self, config):
    self.feature_prediction_module = FeaturePredictionModule(
        config.feature_prediction_module)


def add_vision_decoder_module(self, config):
    convolved_shape = self.self_vision_encoder_module.get_convolved_shape()
    self.vision_decoder_module = VisionDecoderModule(
        config.vision_decoder_module, convolved_shape)


def add_ae_vision_decoder_module(self, config):
    convolved_shape = self.self_vision_encoder_module.get_convolved_shape()
    self.ae_vision_decoder_module = VisionDecoderModule(
        config.vision_decoder_module, convolved_shape)


def add_superposition_module(self, config):
    self.superposition_module = SuperpositionModule(
        config.superposition_module)
    self.rnn_modules.append(self.superposition_module)


def add_motion_generator_module(self, config):
    self.motion_generator_module = MotionGeneratorModule(config.motion_generator_module)
    self.rnn_modules.append(self.motion_generator_module)


class SuperpositionNetworkBase(nn.Module, WithRNNModuleBase):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rnn_modules = []
        add_vision_encoder_module(self, config)
        add_integration_module(self, config)
        add_vision_decoder_module(self, config)
        add_superposition_module(self, config)


class SuperpositionNetwork(SuperpositionNetworkBase):
    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        sm = x['self_motion']

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        om = torch.zeros_like(sm)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, sm, ov_enc, om)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)

        return pred


class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        add_vision_encoder_module(self, config)
        add_ae_vision_decoder_module(self, config)

    def forward(self, sv, decode_from_other=False):

        rec = {}

        sv_enc = self.self_vision_encoder_module(sv)
        rec['self_vision'] = self.ae_vision_decoder_module(sv_enc)

        if decode_from_other:
            ov_enc = self.other_vision_encoder_module(sv)
            rec['other_vision'] = self.ae_vision_decoder_module(ov_enc)

        return rec


class SuperpositionNetworkMotionGeneration(SuperpositionNetworkBase):
    def __init__(self, config):
        super().__init__(config)
        add_motion_generator_module(self, config)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        sm = x['self_motion']

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        om = self.motion_generator_module(ov_enc)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, sm, ov_enc, om)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)
        pred['other_motion'] = om

        return pred


class SuperpositionNetworkFeaturePrediction(SuperpositionNetwork):
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)


class SuperpositionNetworkMotionGenerationFeaturePrediction(SuperpositionNetworkMotionGeneration):
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)
