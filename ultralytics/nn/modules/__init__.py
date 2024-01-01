# coding=utf-8
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .attentions import *
from .block import *
from .conv import *
from .head import *
from .transformer import *

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 'ResNetLayer',

           'RepConv', 'DSConv2D', 'DCNv2', 'ODConv2d', 'SAConv2d', 'ConvAWS2d', 'PConv', 'CoordConv', 'XBNConv',
           'SigConv', 'C3RFEM', 'ASPP', 'SPPFCSPC', 'DiverseBranchBlock', 'C2f_DCN', 'EVCBlock', 'Concat_bifpn',
           'ParNetAttention', 'SE', 'ECA', 'CA', 'CAM', 'GAMAttention', 'DoubleAttention', 'BAMBlock', 'GSConv',
           'EfficientAttention', 'CoTAttention', 'CoordAtt', 'LSKblock', 'MHSA', 'MLCA', 'CARAFE',
           'ParallelPolarizedSelfAttention', 'SpatialGroupEnhance', 'SKAttention', 'SequentialPolarizedSelfAttention',
           'ShuffleAttention', 'SimAM', 'TripletAttention', 'GAMAttention',

           )
