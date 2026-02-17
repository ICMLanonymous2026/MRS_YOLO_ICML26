import math
from typing import List, Tuple

import torch
import torch.nn as nn

# --- vos blocs persos --------------------------------------------------------
from ...nn.convs import Conv
from ...nn.blocks import C3k2

# ──────────────────────────────────────────────────────────────
# 1) Génération du planning de strides anisotropes
# ──────────────────────────────────────────────────────────────
def stride_schedule(start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Renvoie la liste des strides (sH, sW) successifs pour passer de start → target."""
    H, W = start
    h_t, w_t = target
    assert H % h_t == 0 and W % w_t == 0, "Les résolutions doivent être des multiples puissances de 2."

    if H>h_t:
        dh = int(math.log2(H // h_t))  # nombre de /2 sur hauteur
    else: 
        dh = int(math.log2(h_t // H))  # nombre de /2 sur hauteur
    dw = int(math.log2(W // w_t))  # nombre de /2 sur largeurs

    sch = []
    # Axe qui doit downsimpler le plus = on commence par lui seul
    if dh < dw:                         # largeur a plus de /2 à faire
        for _ in range(dw - dh):
            sch.append((1, 2))          # (H inchangé, W /2)
    elif dw < dh:                       # hauteur a plus de /2
        for _ in range(dh - dw):
            sch.append((2, 1))          # (H /2, W inchangé)

    # Puis downsamples synchrones sur les deux axes
    for _ in range(min(dh, dw)):
        sch.append((2, 2))

    return sch  # ex. [(1,2),(1,2),(2,2),(2,2)]


# ──────────────────────────────────────────────────────────────
# 2) Backbone de branche avec strides anisotropes
# ──────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────
# 2) Backbone de branche avec upsample initial si besoin + downsample anisotrope
# ──────────────────────────────────────────────────────────────
class BranchBackbone(nn.Module):
    """
    Rend trois feature maps P3/P4/P5 pour une image d'entrée (H,W).
    Si un upsample est nécessaire pour atteindre la (ou les) dimension(s) cible(s),
    on le fait d'abord en une seule couche vers (max(H, h_t), max(W, w_t)),
    puis on réalise le downsampling tel que précédemment jusqu'à target_hw.
    Les deux premières couches sont des Conv simples, les suivantes des blocs C3k2.
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

        # 0) Upsample initial éventuel vers la dimension la plus grande par axe
        up_H, up_W = max(H, h_t), max(W, w_t)
        layers = []
        i_layer = 0
        self.out_indices = []
        inter_upsample = False

        if (H, W) != (up_H, up_W):
            layers.append(nn.Upsample(size=(up_H, up_W), mode="nearest"))
            i_layer += 1
            inter_upsample = True
            # on indexe cette sortie pour qu'au pire la branche retourne au moins un feature
            self.out_indices.append(i_layer - 1)

        # 1) Planning de downsampling depuis (up_H, up_W) → (h_t, w_t)
        sch = stride_schedule((up_H, up_W), (h_t, w_t))
        L = len(sch)

        c_min = int(64 * width_mult)
        c_final = int(cmax * width_mult)

        if constant_ch:
            ch_val = 128
            channels_list = [ch_val] * L
        else:
            # Calcul des canaux à chaque étape (à rebours)
            channels_list = [c_final] if L > 0 else []
            for _ in range(max(L - 1, 0)):
                prev = min(max(channels_list[-1] // 2, c_min), cmax)
                channels_list.append(prev)
            channels_list.reverse()

        channels = in_ch
        curH = curW = 1
        strides_cumul = []

        # 2) Downsampling anisotrope (identique à avant)
        for i, ((sH, sW), out_ch_i) in enumerate(zip(sch, channels_list)):
            curH *= sH
            curW *= sW
            strides_cumul.append((curH, curW))

            layers.append(Conv(channels, out_ch_i, 3, s=(sH, sW)))
            i_layer += 1

            if i >= 1 and i < 4:
                layers.append(C3k2(out_ch_i, out_ch_i, shortcut=False, c3k=False, n=int(2*depth_mult)))
                i_layer += 1
            elif i >= 4 and i < 6:
                layers.append(C3k2(out_ch_i, out_ch_i, shortcut=True, c3k=False, n=int(2*depth_mult)))
                i_layer += 1
            elif i >= 6:
                layers.append(C3k2(out_ch_i, out_ch_i, shortcut=True, c3k=True, n=int(2*depth_mult)))
                i_layer += 1

            self.out_indices.append(i_layer - 1)
            channels = out_ch_i

        self.model = nn.Sequential(*layers)

        # 3) Strides / out_channels robustes même s'il n'y a que l'upsample
        self.strides = strides_cumul[-3:] if len(strides_cumul) >= 1 else [(1, 1), (1, 1), (1, 1)]
        if L >= 3:
            self._out_channels = tuple(channels_list[-3:])
        elif L >= 1:
            self._out_channels = (channels_list[-1],) * 3
        else:
            # aucun downsample (seulement upsample) → on garde in_ch comme placeholder
            self._out_channels = (in_ch, in_ch, in_ch)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        inter_feats = []
        out = x
        for m in self.model:
            out = m(out)
            inter_feats.append(out)

        # Retourne au moins le dernier tenseur (après upsample si pas de downsample)
        selected_feats = [inter_feats[i] for i in self.out_indices[-3:]] if self.out_indices else [inter_feats[-1]]
        return selected_feats

    def out_channels(self) -> Tuple[int, int, int]:
        return self._out_channels
    

# class SharedKernel(nn.Module):
#     """Poids + biais partagés entre branches, pas de BN ni d’activation."""
#     def __init__(self, in_ch: int, out_ch: int, k: int = 3):
#         super().__init__()
#         self.weight = nn.Parameter(
#             torch.randn(out_ch, in_ch, k, k) / math.sqrt(in_ch * k * k)
#         )
#         self.bias = nn.Parameter(torch.zeros(out_ch))

#     def forward(self, x, stride):
#         return F.conv2d(x, self.weight, self.bias, stride=stride,
#                         padding=k // 2)


# class PartialSharedConv(nn.Module):
#     """
#     x % de canaux proviennent du noyau partagé (poids communs),
#     1-x % de canaux proviennent d’une conv privée.  Chaque flot
#     possède sa BN et son activation pour rester fidèle à Conv.
#     """
#     def __init__(self, kernel_shared: SharedKernel,
#                  in_ch: int, out_ch: int, k_shared: int,
#                  stride: Tuple[int, int]):
#         super().__init__()
#         self.kernel = kernel_shared
#         self.stride = stride

#         # 1-x % privés = conv + BN + Act
#         self.conv_priv = nn.Conv2d(in_ch, out_ch - k_shared, 3,
#                                    stride, padding=1, bias=False)
#         self.bn_priv  = nn.BatchNorm2d(out_ch - k_shared)
#         self.act_priv = nn.SiLU()

#         # BN + Act pour la partie partagée (un par branche)
#         self.bn_sh   = nn.BatchNorm2d(k_shared)
#         self.act_sh  = nn.SiLU()

#     def forward(self, x):
#         # partie partagée
#         xs = self.kernel(x, self.stride)
#         xs = self.act_sh(self.bn_sh(xs))

#         # partie privée
#         xp = self.act_priv(self.bn_priv(self.conv_priv(x)))

#         return torch.cat((xs, xp), 1)



# def build_branches(input_res, width_mult: float = 0.25,
#                    in_ch: int = 1, share_ratio: float = 0.20):

#     def sym_stride(sH, sW):
#         """(2,1) et (1,2) renvoient (1,2) ⇒ clé commune."""
#         return tuple(sorted((sH, sW)))

#     shared_bank: dict[tuple, SharedKernel] = {}   # (layer_idx, sym_stride) → SharedKernel
#     branches = nn.ModuleList()

#     for res in input_res:
#         layers, channels = [], in_ch
#         sch = stride_schedule(res, (16, 16))      # [(sH,sW), …]

#         for idx, (sH, sW) in enumerate(sch):
#             out_ch = max(int(64 * width_mult),
#                          int(1024 * width_mult) // 2 ** (len(sch) - 1 - idx))
#             k_shared = math.ceil(share_ratio * out_ch)
#             key = (idx, sym_stride(sH, sW))

#             if key not in shared_bank:
#                 shared_bank[key] = SharedKernel(channels, k_shared)

#             layers.append(nn.Sequential(
#                 PartialSharedConv(shared_bank[key],
#                                   channels, out_ch, k_shared,
#                                   stride=(sH, sW)),
#                 C3k2(out_ch, out_ch, shortcut=False)
#             ))
#             channels = out_ch

#         branches.append(nn.Sequential(*layers))

#     return branches
