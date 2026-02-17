import torch
import torch.nn as nn
import math

from ..nn.convs import Conv, DWConv
from ..nn.blocks import C3k2, SPPF, C2PSA, DFL
from .base import BaseModel
from ..utils.loss import YOLODetectionLoss
from .Head.detect import Detect


class YOLOv11(BaseModel):
    def __init__(self, output_dir, num_classes=80, strides=[8, 16, 32], reg_max=16, device="cuda:0", input_canals=1, width_mult=0.25, debug=False):
        super().__init__(device=device, output_dir=output_dir)
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.debug = debug

        # scaled channel counts
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)

        # ---------------- Backbone ----------------
        self.conv1 = Conv(input_canals, c1, k=3, s=2)             # P1/2
        self.conv2 = Conv(c1, c2, k=3, s=2)            # P2/4
        self.c3_1 = C3k2(c2, c3, shortcut=False)
        self.conv3 = Conv(c3, c3, k=3, s=2)            # P3/8
        self.c3_2 = C3k2(c3, c3, shortcut=False)
        self.conv4 = Conv(c3, c4, k=3, s=2)            # P4/16
        self.c3_3 = C3k2(c4, c4, shortcut=True)
        self.conv5 = Conv(c4, c5, k=3, s=2)            # P5/32
        self.c3_4 = C3k2(c5, c5, shortcut=True)
        self.sppf = SPPF(c5, c5)
        self.attn = C2PSA(c1=c5, c2=c5, n=2, e=0.5)

        # ---------------- Head (FPN) ----------------
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.head_c3_1 = C3k2(c5 + c4, c4, shortcut=False)  # small P4
        self.head_c3_2 = C3k2(c4 + c3, c3, shortcut=False)  # small P3
        self.down_p3   = Conv(c3, c3, k=3, s=2)
        self.head_c3_3 = C3k2(c3 + c4, c4, shortcut=False)  # medium P4
        self.down_p4   = Conv(c4, c4, k=3, s=2)
        self.head_c3_4 = C3k2(c4 + c5, c5, shortcut=True)   # large P5

        # ---------------- Detect ----------------
        # Using custom Detect with separated branches and DFL
        self.detect = Detect(
            in_channels=[c3, c4, c5],
            num_classes=self.num_classes,
            reg_max=self.reg_max, 
            strides=self.strides
        )

        self.detect.bias_init(image_size=1024)

        # Loss
        self.criterion = YOLODetectionLoss(num_classes=num_classes, strides = self.strides, reg_max=self.reg_max, device=self.device,)
        # self.criterion = SNRYOLODetectionLoss(num_classes=num_classes, strides = self.strides, reg_max=self.reg_max, device=self.device,)

        self.to(self.device)

    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        self.debug_shape("conv1", x)

        x = self.conv2(x)
        self.debug_shape("conv2", x)

        f2 = self.c3_1(x)
        self.debug_shape("c3_1 (f2)", f2)

        x = self.conv3(f2)
        self.debug_shape("conv3", x)

        f3 = self.c3_2(x)
        self.debug_shape("c3_2 (f3)", f3)

        x = self.conv4(f3)
        self.debug_shape("conv4", x)

        f4 = self.c3_3(x)
        self.debug_shape("c3_3 (f4)", f4)

        x = self.conv5(f4)
        self.debug_shape("conv5", x)

        x = self.c3_4(x)
        x = self.sppf(x)
        f5 = self.attn(x)
        self.debug_shape("attn (f5)", f5)

        # Head: small
        p5_up = self.upsample(f5)
        self.debug_shape("p5_up", p5_up)

        p4_feat = torch.cat([p5_up, f4], dim=1)
        self.debug_shape("p4_feat", p4_feat)

        p4_out = self.head_c3_1(p4_feat)
        self.debug_shape("p4_out", p4_out)

        p4_up = self.upsample(p4_out)
        self.debug_shape("p4_up", p4_up)

        p3_feat = torch.cat([p4_up, f3], dim=1)
        self.debug_shape("p3_feat", p3_feat)

        p3_out = self.head_c3_2(p3_feat)
        self.debug_shape("p3_out", p3_out)

        # Head: medium
        p3_down = self.down_p3(p3_out)
        self.debug_shape("p3_down", p3_down)

        pm_feat = torch.cat([p3_down, p4_out], dim=1)
        self.debug_shape("pm_feat", pm_feat)

        p4_out2 = self.head_c3_3(pm_feat)
        self.debug_shape("p4_out2", p4_out2)

        # Head: large
        p4_down = self.down_p4(p4_out2)
        self.debug_shape("p4_down", p4_down)

        pl_feat = torch.cat([p4_down, f5], dim=1)
        self.debug_shape("pl_feat", pl_feat)

        p5_out = self.head_c3_4(pl_feat)
        self.debug_shape("p5_out", p5_out)

        # Detect
        outputs = self.detect(p3_out, p4_out2, p5_out)

        return outputs
    
    def debug_shape(self, name, tensor):
        if self.debug:
            print(f"[DEBUG] {name:<20} shape = {tuple(tensor.shape)}")








