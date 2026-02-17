import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn

# --- vos blocs persos --------------------------------------------------------
from ..nn.convs import Conv
from ..nn.blocks import C3k2, TFSepBlock 
from .base import BaseModel
from ..utils.loss import YOLODetectionLoss
from .Head.detect import Detect
from .Backbones.MR_Backbone import MR_Backbone_F, MR_Backbone_pyramid
from .Backbones.MR_TF_Backbone import MR_Backbone_pyramid as MR_TF_Backbone_pyramid
from .Backbones.MR_TF_Backbone import MR_Backbone_pyramid_upsample as MR_TF_Backbone_pyramid_up


BACKBONE_REGISTRY = {
    'F':     MR_Backbone_F,
    'pyramid':    MR_Backbone_pyramid,
    'TFSep_pyramid': MR_TF_Backbone_pyramid, 
    'TFSep_pyramid_up': MR_TF_Backbone_pyramid_up
}
# -----------------------------------------------------------------------------#
# 3) Modèle complet
# -----------------------------------------------------------------------------#
class MR_YOLO(BaseModel):
    """
    input_resolutions : liste [(H,W), ...] triée librement
    Exemple : [(512,512), (256,1024), (1024,256)]
              ou      [(128,32), (64,64), (32,128)]
    Chaque entrée doit être un **Tensor** (B,C,H,W) fourni dans le même ordre
    à l’appel de `forward`.
    """
    def __init__(self,
                 input_resolutions: List[Tuple[int, int]],
                 output_dir: str,
                 num_classes: int = 80,
                 reg_max: int = 16,
                 device: str = "cuda:0",
                 width_mult: float = 0.25,
                 backbone_mode: str = 'TFSep_pyramid',
                 outfusion_channels_mult: int = 1, 
                 debug: bool = False):
        super().__init__(device=device, output_dir=output_dir)
        self.debug = debug
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.input_resolutions = input_resolutions  

        self.last_forward_features_before_fusion = []
        self.last_forward_features_after_fusion = []

        try:
            BackboneClass = BACKBONE_REGISTRY[backbone_mode]
        except KeyError:
            valid = ', '.join(BACKBONE_REGISTRY.keys())
            raise ValueError(f"Invalid backbone_mode '{backbone_mode}'. "
                            f"Expected one of: {valid}")
        
        self.backbone = BackboneClass(
            input_resolutions      = input_resolutions,
            width_mult             = width_mult,
            out_channels_mult = outfusion_channels_mult,
        )
        c3, c4, c5 = self.backbone.out_channels

        # 2) Neck (FPN/PAN à 3 échelles)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # self.head_c3_1 = C3k2(c5 + c4, c4, shortcut=False)  # petit P4
        self.head_c3_1 = nn.Sequential(
            Conv(c5 + c4, c4, k=1, s=1),
            TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")
        )

        # self.head_c3_2 = C3k2(c4 + c3, c3, shortcut=False)  # petit P3
        self.head_c3_2 = nn.Sequential(
            Conv(c4 + c3, c3, k=1, s=1),
            TFSepBlock(ch=c3, n=1, residual=True, mode="parallel")
        )

        self.down_p3   = Conv(c3, c3, 3, 2)

        # self.head_c3_3 = C3k2(c3 + c4, c4, shortcut=False)  # moyen P4
        self.head_c3_3 = nn.Sequential(
            Conv(c3 + c4, c4, k=1, s=1),
            TFSepBlock(ch=c4, n=1, residual=True, mode="parallel")
        )

        self.down_p4   = Conv(c4, c4, 3, 2)

        # self.head_c3_4 = C3k2(c4 + c5, c5, shortcut=False, c3k=True)   # grand P5
        self.head_c3_4 = nn.Sequential(
            Conv(c4 + c5, c5, k=1, s=1),
            TFSepBlock(ch=c5, n=1, residual=True, mode="parallel")
        )
        # ---------------------------------------------------------------------

        # 3) Déduction des strides
        raw_strides = self.backbone.strides
        strides = [
            max(max(h, w) for (h, w) in (branch[j] for branch in raw_strides))
            for j in range(3)
        ]
        self.strides = strides

        # 4) Détecteur
        self.detect = Detect(
            in_channels=[c3, c4, c5],
            strides=strides,
            num_classes=num_classes,
            reg_max=reg_max
        )

        # 5) Initialisation des biais (taille d'image max logique)
        Hs, Ws = zip(*input_resolutions)
        max_input_dim = max(max(Hs), max(Ws)) 
        self.detect.bias_init(image_size=max_input_dim)

        # 6) Loss
        self.criterion = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.detect.strides,
            reg_max=reg_max,
            device=device
        )

        self.to(device)

    # -------------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------------
    def forward(self, inputs: List[torch.Tensor]):
        # Vérification des dimensions d'entrée
        if len(inputs) != len(self.input_resolutions):
            raise ValueError(f"Nombre d'inputs ({len(inputs)}) ≠ nombre de résolutions "
                             f"({len(self.input_resolutions)})")
        for i, (x, (H_exp, W_exp)) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{i} doit être un tenseur 4D, got {x.dim()}D")
            B, C, H, W = x.shape
            if (H, W) != (H_exp, W_exp):
                raise ValueError(f"Input #{i} a une résolution incorrecte : "
                                 f"attendu (H,W)=({H_exp},{W_exp}), reçu ({H},{W})")

        # 1) Backbone
        p3, p4, p5 = self.backbone(inputs)
        self._dbg("after backbone p3", p3)
        self._dbg("after backbone p4", p4)
        self._dbg("after backbone p5", p5)

        self.last_forward_features_before_fusion = self.backbone.last_forward_features
        self.last_forward_features_after_fusion = [p3, p4, p5]

        # 2) Neck FPN/PAN
        p5_up = self.upsample(p5)
        p4_in = torch.cat([p5_up, p4], dim=1)
        p4_out = self.head_c3_1(p4_in)

        p4_up = self.upsample(p4_out)
        p3_in = torch.cat([p4_up, p3], dim=1)
        p3_out = self.head_c3_2(p3_in)

        p3_d = self.down_p3(p3_out)
        pm_in = torch.cat([p3_d, p4_out], dim=1)
        p4_out2 = self.head_c3_3(pm_in)

        p4_d = self.down_p4(p4_out2)
        pl_in = torch.cat([p4_d, p5], dim=1)
        p5_out = self.head_c3_4(pl_in)

        self._dbg("after Neck p3_out", p3_out)
        self._dbg("after Neck p4_out2", p4_out2)
        self._dbg("after Neck p5_out", p5_out)

        # 3) Détection
        dist_out, cls_out = self.detect(p3_out, p4_out2, p5_out)
        for d, c in zip(dist_out, cls_out):
            self._dbg("after Detect dist_out stats", d)
            self._dbg("after Detect cls_out stats", c)
        return dist_out, cls_out


    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _dbg(self, name, tensor):
        if self.debug:
            print(f"[DBG] {name:<25}: {tuple(tensor.shape)}")