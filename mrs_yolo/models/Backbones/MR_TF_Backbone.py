import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.convs import Conv, DWConv
from ...nn.blocks import TFSepConvs, SPPF, C2PSA, FusionPSA, ChannelAttention, SpatialAttention, PCSA, CBAM, A2C2f, SMSA, SCSA, TFSepBlock, C2f
from ...utils.post_process import non_max_suppression
from .TF_BranchBackbone import BranchBackbone

class MR_Backbone_pyramid(nn.Module):
    def __init__(self,
                 input_resolutions: List[Tuple[int, int]],
                 width_mult: float = 0.25,
                 in_ch: int = 1,
                 out_channels_mult: int = 2,
                 constant_ch: bool = False,
                 fusion_mode: str = "conv"
                 ):
        super().__init__()

        assert fusion_mode in ("conv", "wsum"), "fusion_mode must be 'conv' or 'wsum'"
        self.fusion_mode = fusion_mode

        self.last_forward_features = []

        self.branches = nn.ModuleList([
            BranchBackbone(res, target_hw=(64, 64),
                           width_mult=width_mult,
                           cmax=128,
                           in_ch=in_ch,
                           constant_ch=constant_ch)
            for res in input_resolutions
        ])

        base_ch   = self.branches[0].out_channels()[-1]
        concat_ch = sum(b.out_channels()[-1] for b in self.branches)

        for i, branche in enumerate(self.branches):
            print(f"branche num {i} out channels : {branche.out_channels()}, strides : {branche.strides}, pour dimenson {input_resolutions[i]}")

        if constant_ch:
            c3 = int(constant_ch * width_mult)
            c4 = int(constant_ch * width_mult)
            c5 = int(constant_ch * width_mult)
        else:
            c3 = min(int(1024 * width_mult), base_ch * out_channels_mult)
            c4 = min(int(1024 * width_mult), c3 * 2)
            c5 = min(int(1024 * width_mult), c4 * 2)
        self.out_channels = (c3, c4, c5)
        print('self.out_channels == ', self.out_channels)

        self.strides = []
        for b in self.branches:
            s_h, s_w = b.strides[-1]
            self.strides.append([
                (s_h    , s_w    ),  # P3
                (s_h * 2, s_w * 2),  # P4
                (s_h * 4, s_w * 4),  # P5
            ])

        self.fuse_p3 = nn.Sequential(
                SCSA(channels=concat_ch, with_TF_Attn=True),
                Conv(concat_ch, c3, 1)
            )

        self.c3_p3 = TFSepBlock(c3, n=1, residual=True, mode='parallel')

        # 6) P4 = down(P3) -> TFSepBlock
        self.conv_p4 = Conv(c3, c4, k=3, s=2)
        self.c3_p4   = TFSepBlock(c4, n=1, residual=True, mode='parallel')

        # 7) P5 = down(P4) -> TFSepBlock -> SPPF -> C2PSA
        self.conv_p5 = Conv(c4, c5, k=3, s=2)
        self.c3_p5   = TFSepBlock(c5, n=1, residual=True, mode='parallel')
        self.sppf    = SPPF(c5, c5, k=5)
        self.psa     = C2PSA(c5, c5, n=min(2, c5 // 1024), e=0.5)

    def forward(self, inputs: List[torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1) Last output (P3-level) from each branch
        p3s = [branch(x)[-1] for branch, x in zip(self.branches, inputs)]

        # Keep MR-like bookkeeping
        self.last_forward_features = [tuple(p3s), None, None]

        p3 = self.fuse_p3(torch.cat(p3s, dim=1))  
        p3 = self.c3_p3(p3)

        # 3) Downsamplings to get P4, P5
        p4 = self.conv_p4(p3)
        p4 = self.c3_p4(p4)

        p5 = self.conv_p5(p4)
        p5 = self.c3_p5(p5)
        p5 = self.sppf(p5)
        p5 = self.psa(p5)

        return p3, p4, p5


class MR_Backbone_pyramid_upsample(nn.Module):
    def __init__(self,
                 input_resolutions: List[Tuple[int, int]],
                 width_mult: float = 0.25,
                 in_ch: int = 1, 
                 out_channels_mult = None):
        super().__init__()

        self.last_forward_features = []

        self.branches = nn.ModuleList([
            BranchBackbone(res, target_hw=(1024, 1024),
                           width_mult=width_mult,
                           cmax=64,  
                           in_ch=in_ch,
                           constant_ch=True)  
            for res in input_resolutions
        ])

        concat_ch = len(input_resolutions) * in_ch
        self.strides = [[(16, 16), (32, 32), (64, 64)]] * len(input_resolutions)

        c_last = max(min(int(1024 * width_mult), 2 * concat_ch), int(64 * width_mult))
        self.fuse_1024 = Conv(concat_ch, c_last, 1)

        stem_blocks = []
        for _ in range(4):
            c_next = max(min(int(1024 * width_mult), 2 * c_last), int(64 * width_mult))
            stem_blocks.append(Conv(c_last, c_next, k=3, s=2))
            stem_blocks.append(TFSepBlock(ch=c_next, n=1, residual=True, mode="parallel_fc"))
            c_last = c_next
        self.stem_down = nn.Sequential(*stem_blocks)

        c3 = c_next
        c4 = min(int(1024 * width_mult), c3 * 2)
        c5 = min(int(1024 * width_mult), c4 * 2)
        self.out_channels = (c3, c4, c5)

        # 5) P4 et P5
        self.conv_p4 = Conv(c3, c4, k=3, s=2)
        self.c3_p4 = TFSepBlock(ch=c4, n=1, residual=True, mode="parallel_fc")
        # self.c3_p4 = C2f(c4, c4, n=1, shortcut=True, e=0.5)
        self.conv_p5 = Conv(c4, c5, k=3, s=2)
        self.c3_p5 = TFSepBlock(ch=c5, n=1, residual=True, mode="parallel_fc")
        # self.c3_p5 = C2f(c5, c5, n=1, shortcut=True, e=0.5)
        self.sppf = SPPF(c5, c5, k=5)
        self.psa = C2PSA(c5, c5, n=min(2, c5 // 1024), e=0.5)

    def forward(self, inputs: List[torch.Tensor]):
        feats_1024 = [branch(x)[-1] for branch, x in zip(self.branches, inputs)]
        self.last_forward_features = [tuple(feats_1024), None, None]

        # 2) Fusion
        x = self.fuse_1024(torch.cat(feats_1024, dim=1))

        p3 = self.stem_down(x)
        p4 = self.c3_p4(self.conv_p4(p3))
        p5 = self.psa(self.sppf(self.c3_p5(self.conv_p5(p4))))

        return p3, p4, p5

