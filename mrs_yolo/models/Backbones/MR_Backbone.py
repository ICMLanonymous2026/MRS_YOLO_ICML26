import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn.convs import Conv, DWConv
from ...nn.blocks import C3k2, SPPF, C2PSA, FusionPSA, ChannelAttention, SpatialAttention, PCSA, CBAM, A2C2f, SMSA, SCSA
from ...utils.post_process import non_max_suppression
from .BranchBackbone import BranchBackbone

def make_progressive_fuse_p3(concat_ch: int, c3: int):
    """
    Construit une séquence SCSA + (Conv 1x1) + SCSA + (Conv 1x1) + ...
    qui réduit progressivement les canaux de concat_ch vers c3 en divisant par 2,
    avec TOUJOURS un bloc d'attention juste avant chaque réduction.
    - Si concat_ch == c3 : applique uniquement une SCSA (pas de réduction).
    - Si on doit encore projeter vers c3 après les divisions par 2,
      on ajoute une SCSA avant la projection finale (réduction) vers c3.
    """
    layers = []
    in_ch = concat_ch

    layers.append(SCSA(channels=in_ch))
    while in_ch // 2 >= c3:
        out_ch = in_ch // 2
        layers.append(Conv(in_ch, out_ch, 1))
        in_ch = out_ch
        if in_ch != c3:
            layers.append(SCSA(channels=in_ch))

    if in_ch > c3:
        layers.append(Conv(in_ch, c3, 1))
        in_ch = c3

    if in_ch < c3:
        layers.append(Conv(in_ch, c3, 1))

    return nn.Sequential(*layers)

# class MR_Backbone_pyramid(nn.Module):
#     def __init__(self,
#                  input_resolutions: List[Tuple[int, int]],
#                  width_mult: float = 0.25,
#                  in_ch: int = 1,
#                  out_channels_mult: int = 2,
#                  constant_ch: bool = False,
#                  fusion_mode: str = "conv"
#                  ):
#         super().__init__()

#         assert fusion_mode in ("conv", "wsum"), "fusion_mode must be 'conv' or 'wsum'"
#         self.fusion_mode = fusion_mode

#         # store last forward features like MR_Backbone_F
#         self.last_forward_features = []

#         # 1) Branches multi-résolution (on n'utilisera que le niveau P3 pour concat/sum)
#         self.branches = nn.ModuleList([
#             BranchBackbone(res, target_hw=(64, 64),
#                            width_mult=width_mult,
#                            cmax=256,
#                            in_ch=in_ch,
#                            constant_ch=constant_ch)
#             for res in input_resolutions
#         ])

#         base_ch   = self.branches[0].out_channels()[-1]
#         concat_ch = sum(b.out_channels()[-1] for b in self.branches)

#         for i, branche in enumerate(self.branches):
#             print(f"branche num {i} out channels : {branche.out_channels()}, strides : {branche.strides}, pour dimenson {input_resolutions[i]}")

#         # 3) Définition des c3, c4, c5 (pyramide)
#         if constant_ch:
#             c3 = int(constant_ch * width_mult)
#             c4 = int(constant_ch * width_mult)
#             c5 = int(constant_ch * width_mult)
#         else:
#             c3 = min(int(1024 * width_mult), base_ch * out_channels_mult)
#             c4 = min(int(1024 * width_mult), c3 * 2)
#             c5 = min(int(1024 * width_mult), c4 * 2)
#         self.out_channels = (c3, c4, c5)
#         print('self.out_channels == ', self.out_channels)

#         # 4) Strides exportés
#         self.strides = []
#         for b in self.branches:
#             s_h, s_w = b.strides[-1]
#             self.strides.append([
#                 (s_h    , s_w    ),  # P3
#                 (s_h * 2, s_w * 2),  # P4
#                 (s_h * 4, s_w * 4),  # P5
#             ])

#         # ---------- P3 fusion choices ----------
#         if self.fusion_mode == "conv":
#             # P3 = concat -> 1x1 -> C3k2
#             self.fuse_p3 = nn.Sequential(
#                 make_progressive_fuse_p3(concat_ch, c3)
#             )
#         else:
#             # "wsum": per-branch 1x1 adapters to c3 + softmax weights
#             branch_out_ch = [b.out_channels()[-1] for b in self.branches]
#             self.p3_adapters = nn.ModuleList([
#                 nn.Identity() if ch == c3 else Conv(ch, c3, 1)
#                 for ch in branch_out_ch
#             ])
#             # learnable scalar weights (one per branch)
#             self.p3_weights = nn.Parameter(torch.zeros(len(self.branches)))  # initialized equal after softmax
#             # no fuse_p3 conv here; we directly form the weighted sum into c3

#         self.c3_p3 = C3k2(c3, c3, shortcut=True, n=1, c3k=True)

#         # 6) P4 = down(P3) -> C3k2
#         self.conv_p4 = Conv(c3, c4, k=3, s=2)
#         self.c3_p4   = C3k2(c4, c4, shortcut=True, n=1, c3k=True)

#         # 7) P5 = down(P4) -> C3k2 -> SPPF -> C2PSA
#         self.conv_p5 = Conv(c4, c5, k=3, s=2)
#         self.c3_p5   = C3k2(c5, c5, shortcut=True, n=1, c3k=True)
#         self.sppf    = SPPF(c5, c5, k=5)
#         self.psa     = C2PSA(c5, c5, n=min(2, c5 // 1024), e=0.5)

#     def forward(self, inputs: List[torch.Tensor]
#                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         # 1) Last output (P3-level) from each branch
#         p3s = [branch(x)[-1] for branch, x in zip(self.branches, inputs)]

#         # Keep MR-like bookkeeping
#         self.last_forward_features = [tuple(p3s), None, None]

#         # 2) Fusion P3
#         if self.fusion_mode == "conv":
#             p3 = self.fuse_p3(torch.cat(p3s, dim=1))  # (B, c3, 64, 64)
#         else:
#             # adapt each to c3 then softmax-weighted sum
#             adapted = [adapt(t) for adapt, t in zip(self.p3_adapters, p3s)]
#             alphas = F.softmax(self.p3_weights, dim=0)  # (num_branches,)
#             # Broadcast weights to (B, C, H, W)
#             p3 = sum(a * w.view(1, 1, 1, 1) for a, w in zip(adapted, alphas))

#         p3 = self.c3_p3(p3)

#         # 3) Downsamplings to get P4, P5
#         p4 = self.conv_p4(p3)
#         p4 = self.c3_p4(p4)

#         p5 = self.conv_p5(p4)
#         p5 = self.c3_p5(p5)
#         p5 = self.sppf(p5)
#         p5 = self.psa(p5)

#         return p3, p4, p5
    

class MR_Backbone_pyramid(nn.Module):
    def __init__(self,
                 input_resolutions: List[Tuple[int, int]],
                 width_mult: float = 0.25,
                 in_ch: int = 1,
                 out_channels_mult: int = 2,
                 constant_ch: bool = False):
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

        c_last = max(min(int(1024 * width_mult), 2*concat_ch), int(64*width_mult))
        self.fuse_1024 = Conv(concat_ch, c_last, 1)
        stem_blocks = []
        for _ in range(4):
            c_next = max(min(int(1024 * width_mult), 2*c_last), int(64*width_mult))
            stem_blocks.append(Conv(c_last, c_next, k=3, s=2))
            stem_blocks.append(C3k2(c_next, c_next, shortcut=True, n=1, c3k=True))
            c_last = c_next
        self.stem_down = nn.Sequential(*stem_blocks)

        c3 = c_next
        c4 = min(int(1024 * width_mult), c3 * 2)
        c5 = min(int(1024 * width_mult), c4 * 2)
        self.out_channels = (c3, c4, c5)

        self.conv_p4 = Conv(c3, c4, k=3, s=2)
        self.c3_p4   = C3k2(c4, c4, shortcut=True, n=1, c3k=True)
        self.conv_p5 = Conv(c4, c5, k=3, s=2)
        self.c3_p5   = C3k2(c5, c5, shortcut=True, n=1, c3k=True)
        self.sppf    = SPPF(c5, c5, k=5)
        self.psa     = C2PSA(c5, c5, n=min(2, c5 // 1024), e=0.5)

    def forward(self, inputs: List[torch.Tensor]):
        feats_1024 = [branch(x)[-1] for branch, x in zip(self.branches, inputs)]
        self.last_forward_features = [tuple(feats_1024), None, None]

        x = self.fuse_1024(torch.cat(feats_1024, dim=1))

        p3 = self.stem_down(x)
        p4 = self.c3_p4(self.conv_p4(p3))
        p5 = self.psa(self.sppf(self.c3_p5(self.conv_p5(p4))))

        return p3, p4, p5



class MR_Backbone_F(nn.Module):
    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        width_mult: float = 0.25,
        in_ch: int = 1,
        out_channels_mult: int = 3,
        constant_ch: bool = False,
    ):
        super().__init__()
        self.out_channels_mult = out_channels_mult
        self.num_branches = len(input_resolutions)
        self.last_forward_features = []

        self.branches = nn.ModuleList([
            BranchBackbone(
                res, (16, 16),
                width_mult=width_mult,
                in_ch=in_ch,
                constant_ch=constant_ch
            ) for res in input_resolutions
        ])
        self.strides = [b.strides for b in self.branches]

        c3, c4, c5 = self.branches[0].out_channels()
        self.out_channels = tuple(c * out_channels_mult for c in (c3, c4, c5))
        c_sums = [c * len(self.branches) for c in (c3, c4, c5)]

        self.projections = nn.ModuleList([
            nn.Sequential(  # P3
                # CBAM(c_sums[0]),  
                # C2PSA(c_sums[0], c_sums[0],
                #     n=min(2, c_sums[2] // 1024), e=0.5),  
                # PCSA(c_sums[0]),     
                # A2C2f(c_sums[0], c_sums[0], n=min(2, self.out_channels[0] // 1024), a2=True, area=1, residual=True),   
                Conv(c_sums[0], self.out_channels[0], 1),
                C3k2(self.out_channels[0], self.out_channels[0],
                     shortcut=True, n=min(2, self.out_channels[0] // 1024), c3k=True)
            ),
            nn.Sequential(  # P4
                # CBAM(c_sums[1]),  
                # C2PSA(c_sums[1], c_sums[1],
                #     n=min(2, c_sums[2] // 1024), e=0.5),    
                # PCSA(c_sums[1]),   
                # A2C2f(c_sums[1], c_sums[1], n=min(2, self.out_channels[1] // 1024), a2=True, area=1, residual=True),  
                Conv(c_sums[1], self.out_channels[1], 1),
                C3k2(self.out_channels[1], self.out_channels[1],
                     shortcut=True, n=min(2, self.out_channels[1] // 1024), c3k=True)
            ),
            # nn.Sequential(  # P5
            #     Conv(c_sums[2], self.out_channels[2], 1),
            #     SPPF(self.out_channels[2], self.out_channels[2], k=5),
            #     C2PSA(self.out_channels[2], self.out_channels[2],
            #           n=min(2, self.out_channels[2] // 1024), e=0.5)
            # )
            nn.Sequential(  # P5
                SPPF(c_sums[2], c_sums[2], k=5),
                C2PSA(c_sums[2], c_sums[2],
                    n=min(2, c_sums[2] // 1024), e=0.5),
                # PCSA(c_sums[2]),
                # CBAM(c_sums[2]),     
                # A2C2f(c_sums[2], c_sums[2], n=min(2, self.out_channels[2] // 1024), a2=True, area=1, residual=True),      
                Conv(c_sums[2], self.out_channels[2], 1),
                C3k2(self.out_channels[2], self.out_channels[2], n=min(2, self.out_channels[2] // 1024), c3k=True)  
            )
        ])

    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3s, p4s, p5s = zip(*[branch(x) for branch, x in zip(self.branches, inputs)])
        self.last_forward_features = [p3s, p4s, p5s]
        p3_cat = torch.cat(p3s, 1)
        p4_cat = torch.cat(p4s, 1)
        p5_cat = torch.cat(p5s, 1)
        return (
            self.projections[0](p3_cat),
            self.projections[1](p4_cat),
            self.projections[2](p5_cat)
        )




