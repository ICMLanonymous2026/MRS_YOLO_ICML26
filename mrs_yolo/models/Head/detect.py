
import torch
import torch.nn as nn
import math

from ...nn.convs import Conv, DWConv
from ...nn.blocks import DFL


class Detect(nn.Module):
    """
    Custom YOLO Detect head with separated bbox and cls/objectness branches and DFL decoding:
    - in_channels: list of channel dims for P3, P4, P5 feature maps
    - num_classes: number of object classes (0 → only objectness)
    - anchors_per_level: number of anchors per spatial location
    - reg_max: number of bins for bbox distribution
    """
    def __init__(self, in_channels, strides, num_classes=80, reg_max=16):
        super().__init__()
        self.nc = num_classes                   # number of classes
        self.reg_max = reg_max                  # distribution bins per bbox side
        self.nl = len(in_channels)              # number of detection layers
        self.legacy=True
        self.strides = strides

        # Determine channels for classification branch:
        c2, c3 = max((16, in_channels[0] // 4, self.reg_max * 4)), max(in_channels[0], min(self.nc, 100))  # channels

        # Branch for bbox distribution regression (4 sides * reg_max bins)
        self.cv_dist = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in in_channels
        )

        self.cv_clsobj = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in in_channels)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in in_channels
            )
        )

        # Distribution Focal Loss decoder for bbox
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, *features):
        # features: P3, P4, P5
        dist_outputs = [conv(f) for conv, f in zip(self.cv_dist, features)]
        clsobj_outputs = [conv(f) for conv, f in zip(self.cv_clsobj, features)]
        return dist_outputs, clsobj_outputs
    
    def bias_init(self, image_size=640):
        """
        Initialise les biais des têtes bbox (DFL) et classification pour faciliter la convergence.
        """
        for a, b, s in zip(self.cv_dist, self.cv_clsobj, self.strides):
            # Initialisation des biais de la branche bbox (DFL)
            if hasattr(a[-1], "bias"):
                a[-1].bias.data[:] = 1.0

            # Initialisation des biais de la branche classification
            if hasattr(b[-1], "bias"):
                b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (image_size / s) ** 2)

