import math
from typing import List, Tuple

import torch
import torch.nn as nn

from ...nn.convs import Conv
from ...nn.blocks import TFSepBlock, C2f                   


def stride_schedule(start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Renvoie la liste des strides (sH, sW) successifs pour passer de start → target."""
    H, W = start
    h_t, w_t = target
    assert H % h_t == 0 and W % w_t == 0, "Les résolutions doivent être des multiples puissances de 2."

    if H > h_t:
        dh = int(math.log2(H // h_t))  
    else:
        dh = int(math.log2(h_t // H)) 
    dw = int(math.log2(W // w_t))      

    sch = []
    if dh < dw:                         
        for _ in range(dw - dh):
            sch.append((1, 2))          
    elif dw < dh:                      
        for _ in range(dh - dw):
            sch.append((2, 1))          
    for _ in range(min(dh, dw)):
        sch.append((2, 2))

    return sch  


class BranchBackbone(nn.Module):
    """
    Rend trois feature maps P3/P4/P5 pour une image d'entrée (H,W).
    Si un upsample est nécessaire pour atteindre la (ou les) dimension(s) cible(s),
    on le fait d'abord en une seule couche vers (max(H, h_t), max(W, w_t)),
    puis on réalise le downsampling tel que précédemment jusqu'à target_hw.
    Les deux premières couches sont des Conv simples, les suivantes des blocs TFSepBlock.
    """
    def __init__(self,
                 start_hw: Tuple[int, int],
                 target_hw: Tuple[int, int],
                 width_mult: float = 0.25,
                 depth_mult: float = 0.5,
                 in_ch: int = 1,
                 cmax: int = 1024,
                 constant_ch: bool = False):
        super().__init__()
        H, W = start_hw
        h_t, w_t = target_hw

        up_H, up_W = max(H, h_t), max(W, w_t)
        layers = []
        i_layer = 0
        self.out_indices = []

        if (H, W) != (up_H, up_W):
            layers.append(nn.Upsample(size=(up_H, up_W), mode="nearest"))
            i_layer += 1
            self.out_indices.append(i_layer - 1)

        sch = stride_schedule((up_H, up_W), (h_t, w_t))
        L = len(sch)

        c_min = int(64 * width_mult)
        c_final = int(cmax * width_mult)

        if constant_ch:
            ch_val = 128
            channels_list = [ch_val] * L
        else:
            channels_list = [c_final] if L > 0 else []
            for _ in range(max(L - 1, 0)):
                prev = min(max(channels_list[-1] // 2, c_min), cmax)
                channels_list.append(prev)
            channels_list.reverse()

        channels = in_ch
        curH = curW = 1
        strides_cumul = []

        for i, ((sH, sW), out_ch_i) in enumerate(zip(sch, channels_list)):
            curH *= sH
            curW *= sW
            strides_cumul.append((curH, curW))

            layers.append(Conv(channels, out_ch_i, 3, s=(sH, sW)))
            i_layer += 1

            n_tf = max(1, int(2 * depth_mult))  
            if 1 <= i < 4:
                residual = False   
            elif 4 <= i < 6:
                residual = True    
            else: 
                residual = True   
                n_tf = n_tf + 1   

            layers.append(
                TFSepBlock(
                ch=out_ch_i, n=n_tf, residual=residual,
                k_f=3, k_t=3, shuffle_groups=2, mode='parallel'
                )
            )
            i_layer += 1

            self.out_indices.append(i_layer - 1)
            channels = out_ch_i

        self.model = nn.Sequential(*layers)

        self.strides = strides_cumul[-3:] if len(strides_cumul) >= 1 else [(1, 1), (1, 1), (1, 1)]
        if L >= 3:
            self._out_channels = tuple(channels_list[-3:])
        elif L >= 1:
            self._out_channels = (channels_list[-1],) * 3
        else:
            self._out_channels = (in_ch, in_ch, in_ch)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        inter_feats = []
        out = x
        for m in self.model:
            out = m(out)
            inter_feats.append(out)

        selected_feats = [inter_feats[i] for i in self.out_indices[-3:]] if self.out_indices else [inter_feats[-1]]
        return selected_feats

    def out_channels(self) -> Tuple[int, int, int]:
        return self._out_channels

