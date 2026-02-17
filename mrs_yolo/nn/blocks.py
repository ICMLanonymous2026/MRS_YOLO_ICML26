# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .convs import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
# from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initialize the YOLOv8 mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """
        Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """
        Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """
        Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (Tuple[int, int, int]): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """
        Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """
        Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


# class C3TR(C3):
#     """C3 module with TransformerBlock()."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """
#         Initialize C3 module with TransformerBlock.

#         Args:
#             c1 (int): Input channels.
#             c2 (int): Output channels.
#             n (int): Number of Transformer blocks.
#             shortcut (bool): Whether to use shortcut connections.
#             g (int): Groups for convolutions.
#             e (float): Expansion ratio.
#         """
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """
        Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """
        Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """
        Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """
        Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """
        Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """
        Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """
        Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """
        Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """
        Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """
        Forward pass of ImagePoolingAttn.

        Args:
            x (List[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """
        Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """
        Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    def forward_fuse(self, x, w):
        """
        Passes input out unchanged.

        TODO: Update or remove?
        """
        return x

    def forward(self, x, w):
        """
        Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """
        Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """
        Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """
        Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """
        Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """
        Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """
        Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """
        Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (List[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """
        Initialize CBFuse module.

        Args:
            idx (List[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """
        Forward pass through CBFuse layer.

        Args:
            xs (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """
        Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """
        Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """
        Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """
        Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use local key connection.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """
        Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """
        Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """
        Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """
        Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class FusionPSA(nn.Module):
    """
    FusionPSA module for attention-based feature fusion with channel reduction.

    This module is designed to process concatenated multi-branch features by:
    1. Expanding and splitting the input into two streams (residual + attention).
    2. Applying one or more PSABlock layers on the attention stream.
    3. Fusing both streams and reducing the number of output channels.

    This is useful for intelligent multi-scale feature fusion while controlling
    channel dimensionality.

    Args:
        c_in (int): Number of input channels (e.g., sum of branch outputs).
        c_out (int): Number of desired output channels.
        n (int): Number of PSABlock layers to apply (default: 1).
        e (float): Expansion ratio for internal attention channels (default: 0.5).

    Attributes:
        cv1 (Conv): 1x1 convolution to expand input to 2 * c_int channels (for residual + attention split).
        cv2 (Conv): 1x1 convolution to reduce fused features to c_out channels.
        blocks (nn.Sequential): Stack of PSABlock modules operating on attention stream.

    Example:
        >>> fusion = FusionPSA(c_in=320, c_out=96, n=2)
        >>> x = torch.randn(1, 320, 32, 32)
        >>> y = fusion(x)  # Output shape: (1, 96, 32, 32)
    """
    def __init__(self, c_in: int, c_out: int, n: int = 1, e: float = 0.5):
        super().__init__()
        c_int = int(c_out * e)
        self.cv1 = Conv(c_in, 2 * c_int, 1)
        self.cv2 = Conv(2 * c_int, c_out, 1)
        self.blocks = nn.Sequential(*[
            PSABlock(c=c_int, num_heads=max(1, c_int // 64), attn_ratio=0.5)
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FusionPSA.

        Args:
            x (torch.Tensor): Input tensor of shape (B, c_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, c_out, H, W).
        """
        a, b = self.cv1(x).chunk(2, dim=1)
        b = self.blocks(b)
        return self.cv2(torch.cat((a, b), dim=1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """
        Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """
        Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """
        Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | List[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """
        Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """
        Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc, ec, e=4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor."""
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m) -> None:
        """Initialize residual module with the wrapped module."""
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x):
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch, c3, embed):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension."""
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
    

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class PCSA(nn.Module):
    def __init__(self, channels, pool_kernel=7):
        """
        Args:
            channels (int): Number of input/output channels.
            pool_kernel (int): Kernel size for initial average pooling (e.g., 7).
        """
        super(PCSA, self).__init__()
        self.channels = channels
        self.pool_kernel = pool_kernel

        # Projections for Q, K, V (DWConv1d with kernel size 1)
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1, groups=channels)

        # Final attention pooling + sigmoid
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Attention-weighted output of shape [B, C, H, W]
        """
        B, C, H, W = x.size()

        # Step 1: Avg pooling to reduce spatial dimensions
        x_pool = F.avg_pool2d(x, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        _, _, H_, W_ = x_pool.size()
        N = H_ * W_  # N = number of spatial tokens

        # Step 2: Reshape to [B, C, N]
        x_p = x_pool.view(B, C, N)

        # Step 3: Q, K, V projections
        Q = self.q_proj(x_p)  # [B, C, N]
        K = self.k_proj(x_p)  # [B, C, N]
        V = self.v_proj(x_p)  # [B, C, N]

        # Step 4: Attention: softmax(Q K^T / sqrt(C)) V
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (C ** 0.5)  # [B, C, C]
        attn_weights = F.softmax(attn_scores, dim=-1)
        x_attn = torch.bmm(attn_weights, V)  # [B, C, N]

        # Step 5: Reshape and average pool to [B, C, 1, 1]
        x_attn = x_attn.view(B, C, H_, W_)
        x_attn_pooled = self.output_pool(x_attn)  # [B, C, 1, 1]

        # Step 6: Apply sigmoid and channel-wise scaling
        scale = self.sigmoid(x_attn_pooled)  # [B, C, 1, 1]
        out = x * scale

        return out


class SMSA(nn.Module):
    """
    Shared Multi-Semantic Spatial Attention (SMSA)

    Entrée :  x in (B, C, H, W)
    Sortie :  x * AttnH * AttnW  (mêmes dimensions)
    - AttnH est appris à partir de X_H = GAP_H(x) -> (B, C, W)
    - AttnW est appris à partir de X_W = GAP_W(x) -> (B, C, H)
    - Canaux C sont divisés en K sous-groupes (C doit être divisible par K).
    - Chaque sous-groupe est filtré avec une DW-Conv1d (k_i) pour capter des
      sémantiques multi-échelles, puis passé par une conv1d "align" partagée
      entre les voies H et W pour lier implicitement les deux décompositions.
    - Concat -> GroupNorm(K groupes) -> Sigmoid pour produire les attentions.
    """

    def __init__(self, channels: int, K: int = 4, kernel_sizes = [3, 5, 7, 9]):
        super().__init__()
        assert channels % K == 0, "channels must be divisible by K"
        self.C = channels
        self.K = K
        self.c_sub = channels // K

        # Noyaux par sous-feature (si K != len(kernel_sizes), on cycle)
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]
        self.ks = kernel_sizes

        # Convolutions depthwise 1D par sous-groupe pour la voie H et W.
        # On utilise groups=c_sub pour du depthwise au sein du sous-groupe.
        self.dw_convs_H = nn.ModuleList()
        self.dw_convs_W = nn.ModuleList()
        for i in range(K):
            k = self.ks[i % len(self.ks)]
            pad = k // 2
            self.dw_convs_H.append(
                nn.Conv1d(self.c_sub, self.c_sub, kernel_size=k, padding=pad, groups=self.c_sub, bias=False)
            )
            self.dw_convs_W.append(
                nn.Conv1d(self.c_sub, self.c_sub, kernel_size=k, padding=pad, groups=self.c_sub, bias=False)
            )

        # Conv1d "align" partagée entre H et W (mêmes poids utilisés pour les deux voies),
        # pour mixer légèrement les canaux au sein de chaque sous-groupe (groups=1).
        self.align_conv = nn.Conv1d(self.c_sub, self.c_sub, kernel_size=1, bias=False)

        # Normalisations GroupNorm à K groupes pour chaque voie (sur la dimension canaux)
        self.gn_H = nn.GroupNorm(num_groups=K, num_channels=channels, affine=True)
        self.gn_W = nn.GroupNorm(num_groups=K, num_channels=channels, affine=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.C, "Input channels must match 'channels' given at init."

        # Séquences 1D par pooling global sur l'autre dimension
        # X_H: (B, C, W)    (moyenne sur H)
        # X_W: (B, C, H)    (moyenne sur W)
        X_H = x.mean(dim=2)  # (B, C, W)
        X_W = x.mean(dim=3)  # (B, C, H)

        # Découpage en K sous-groupes et filtrage DW-Conv1d + align partagé
        outs_H = []
        outs_W = []
        for i in range(self.K):
            c0 = i * self.c_sub
            c1 = (i + 1) * self.c_sub

            Xi_H = X_H[:, c0:c1, :]  # (B, c_sub, W)
            Xi_W = X_W[:, c0:c1, :]  # (B, c_sub, H)

            # Depthwise conv 1D (multi-semantic via kernels variés)
            Yi_H = self.dw_convs_H[i](Xi_H)  # (B, c_sub, W)
            Yi_W = self.dw_convs_W[i](Xi_W)  # (B, c_sub, H)

            # Conv1d "align" partagée entre H et W
            Yi_H = self.align_conv(Yi_H)     # (B, c_sub, W)
            Yi_W = self.align_conv(Yi_W)     # (B, c_sub, H)  -- mêmes poids

            outs_H.append(Yi_H)
            outs_W.append(Yi_W)

        # Concat sur les canaux -> (B, C, W) et (B, C, H)
        Y_H = torch.cat(outs_H, dim=1)
        Y_W = torch.cat(outs_W, dim=1)

        # GroupNorm à K groupes (meilleure séparation sémantique que BN), puis Sigmoid
        AttnH_seq = self.sigmoid(self.gn_H(Y_H))  # (B, C, W)
        AttnW_seq = self.sigmoid(self.gn_W(Y_W))  # (B, C, H)

        # Remonter en 2D par broadcast
        AttnH = AttnH_seq.unsqueeze(2)         # (B, C, 1, W)  -> broadcast sur H
        AttnW = AttnW_seq.unsqueeze(3)         # (B, C, H, 1)  -> broadcast sur W

        return x * AttnH * AttnW


class SCSA(nn.Module):
    """
    SCSA: Sequential (S) application of Shared Multi-Semantic Spatial Attention (SMSA)
          then Progressive Channel-wise Self-Attention (PCSA).

    Pipeline: x --SMSA--> (spatial attention) --PCSA--> (channel attention)
              --> (optionnel) résiduel x + gamma * y

    Args:
        channels (int): nombre de canaux d'entrée/sortie.
        K (int): nombre de sous-groupes pour SMSA (C doit être divisible par K).
        sms_kernel_sizes (list[int]): noyaux DWConv1d multi-sémantiques pour SMSA.
        pcsa_pool_kernel (int): taille de pooling spatial initiale dans PCSA.
        residual (bool): active le lien résiduel.
        gamma_init (float): initialisation de l'échelle résiduelle (0.0 recommandé).

    Pré-requis:
        - Les classes `SMSA` et `PCSA` doivent être définies (celles que tu as fournies).
    """

    def __init__(self,
                 channels: int,
                 K: int = 4,
                 sms_kernel_sizes = [3, 5, 7, 9],
                 pcsa_pool_kernel: int = 7,
                 residual: bool = True,
                 gamma_init: float = 0.0,
                 with_TF_Attn=False):
        super().__init__()
        self.residual = residual

        if with_TF_Attn:
            self.smsa = TFSepBlock(ch=channels, mode='parallel')
        else:
            self.smsa = SMSA(channels=channels, K=K, kernel_sizes=sms_kernel_sizes)
        self.pcsa = PCSA(channels=channels, pool_kernel=pcsa_pool_kernel)

        if residual:
            # échelle résiduelle apprenable (style "gamma" pour stabiliser l'entraînement)
            self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        else:
            self.register_parameter('gamma', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.smsa(x)   # spatial attention (H/W)
        y = self.pcsa(y)   # channel attention
        if self.residual:
            return x + self.gamma * y
        return y


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """
    ShuffleNet-style channel shuffle.
    x: (B, C, F, T)
    """
    B, C, F_, T_ = x.shape
    assert C % groups == 0, "Le nombre de canaux doit être divisible par 'groups'."
    x = x.view(B, groups, C // groups, F_, T_)
    x = x.transpose(1, 2).contiguous()
    x = x.view(B, C, F_, T_)
    return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1, 1), s=(1, 1), p=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    
class TFSepConvs(nn.Module):
    """
    Time-Frequency Separate Convolutions (TF-SepConvs)

    Entrée :  (B, C, F, T)
    Sortie :  (B, C_out, F, T) avec C_out = out_channels

    Étapes :
      1) Transition 1x1: C -> out_channels
      2) Channel shuffle
      3) Split en deux moitiés (fréquence / temps)
      4) Voie F : depthwise (3x1) -> avg(F) -> pointwise 1x1 -> broadcast+add
      5) Voie T : depthwise (1x3) -> avg(T) -> pointwise 1x1 -> broadcast+add
      6) Concat canal
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_f: int = 3,            # kernel le long de F (3x1)
        k_t: int = 3,            # kernel le long de T (1x3)
        shuffle_groups: int = 2
    ):
        super().__init__()
        assert out_channels % 2 == 0, "out_channels doit être pair (division en deux voies)."
        self.out_channels = out_channels
        self.half = out_channels // 2
        self.shuffle_groups = shuffle_groups

        # 1) Transition 1x1
        self.transition = ConvBNReLU(in_channels, out_channels, k=(1, 1))

        # 4) Chemin fréquentiel (depthwise 3x1)
        pad_f = k_f // 2
        self.dw_f = ConvBNReLU(self.half, self.half, k=(k_f, 1), p=(pad_f, 0),
                               groups=self.half)  # depthwise
        self.pw_f = ConvBNReLU(self.half, self.half, k=(1, 1))

        # 5) Chemin temporel (depthwise 1x3)
        pad_t = k_t // 2
        self.dw_t = ConvBNReLU(self.half, self.half, k=(1, k_t), p=(0, pad_t),
                               groups=self.half)  # depthwise
        self.pw_t = ConvBNReLU(self.half, self.half, k=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)
        """
        # 1) Transition
        y = self.transition(x)  # (B, C', F, T)

        # 2) Channel shuffle
        y = channel_shuffle(y, groups=self.shuffle_groups)

        # 3) Split
        y_f, y_t = torch.split(y, self.half, dim=1)  # (B, C'/2, F, T) chacun

        # 4) Voie F
        z_f = self.dw_f(y_f)                     # depthwise 3x1
        v_f = z_f.mean(dim=2, keepdim=True)      # avg over F -> (B, C'/2, 1, T)
        v_f = self.pw_f(v_f)                     # 1x1 -> (B, C'/2, 1, T)
        xhat_f = y_f + v_f                       # broadcast add sur F

        # 5) Voie T
        z_t = self.dw_t(y_t)                     # depthwise 1x3
        v_t = z_t.mean(dim=3, keepdim=True)      # avg over T -> (B, C'/2, F, 1)
        v_t = self.pw_t(v_t)                     # 1x1 -> (B, C'/2, F, 1)
        xhat_t = y_t + v_t                       # broadcast add sur T

        # 6) Concat canaux
        out = torch.cat([xhat_f, xhat_t], dim=1)  # (B, C', F, T)

        return out
    

class TFParallelDWConv(nn.Module):
    """
    Parallel TF block without pre-splitting channels.

    Input:  (B, C, F, T)
    Output: (B, C, F, T)  [concat of two (C/2)-channel branches]

    Path F (frequency-focused):
      - Depthwise (k_f x 1), groups=C
      - Pointwise 1x1: C -> C/2
      - Axis-wise context: avg over F -> 1x1 -> broadcast add

    Path T (time-focused):
      - Depthwise (1 x k_t), groups=C
      - Pointwise 1x1: C -> C/2
      - Axis-wise context: avg over T -> 1x1 -> broadcast add
    """
    def __init__(self, channels: int, k_f: int = 3, k_t: int = 3):
        super().__init__()
        assert channels % 2 == 0, "channels must be even (C/2 + C/2)."
        self.channels = channels
        self.half = channels // 2

        # Frequency path
        self.dw_f = ConvBNReLU(
            in_ch=channels,
            out_ch=channels,     # depthwise keeps C (groups=C)
            k=(k_f, 1),
            p=(k_f // 2, 0),
            groups=channels
        )
        self.pw_f = ConvBNReLU(channels, self.half, k=(1, 1))
        self.ctx_f = ConvBNReLU(self.half, self.half, k=(1, 1))  # context 1x1

        # Time path
        self.dw_t = ConvBNReLU(
            in_ch=channels,
            out_ch=channels,
            k=(1, k_t),
            p=(0, k_t // 2),
            groups=channels
        )
        self.pw_t = ConvBNReLU(channels, self.half, k=(1, 1))
        self.ctx_t = ConvBNReLU(self.half, self.half, k=(1, 1))  # context 1x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape

        # Frequency path
        yf = self.dw_f(x)                 # (B, C, F, T)
        yf = self.pw_f(yf)                # (B, C/2, F, T)
        vf = yf.mean(dim=2, keepdim=True) # (B, C/2, 1, T)
        vf = self.ctx_f(vf)               # (B, C/2, 1, T)
        yf = yf + vf                      # broadcast over F

        # Time path
        yt = self.dw_t(x)                 # (B, C, F, T)
        yt = self.pw_t(yt)                # (B, C/2, F, T)
        vt = yt.mean(dim=3, keepdim=True) # (B, C/2, F, 1)
        vt = self.ctx_t(vt)               # (B, C/2, F, 1)
        yt = yt + vt                      # broadcast over T

        # Concat back to C channels
        out = torch.cat([yf, yt], dim=1)  # (B, C, F, T)
        return out


class TFParallelDWSum(nn.Module):
    """
    Parallel TF block WITHOUT channel reduction (C -> C), then SUM the two branches.

    Input:  (B, C, F, T)
    Output: (B, C, F, T)

    Path F (frequency-focused):
      - Depthwise (k_f x 1), groups=C
      - Pointwise 1x1: C -> C
      - Axis-wise context: avg over F -> 1x1 -> broadcast add

    Path T (time-focused):
      - Depthwise (1 x k_t), groups=C
      - Pointwise 1x1: C -> C
      - Axis-wise context: avg over T -> 1x1 -> broadcast add

    Combine:
      - out = yf + yt
      (Optionally scale by a factor if needed; BN usually stabilizes.)
    """
    def __init__(self, channels: int, k_f: int = 3, k_t: int = 3, sum_scale: float = 1.0):
        super().__init__()
        assert channels > 0
        self.channels = channels
        self.sum_scale = sum_scale

        # Frequency path
        self.dw_f = ConvBNReLU(
            in_ch=channels,
            out_ch=channels,          # depthwise keeps C (groups=C)
            k=(k_f, 1),
            p=(k_f // 2, 0),
            groups=channels
        )
        self.pw_f  = ConvBNReLU(channels, channels, k=(1, 1))
        self.ctx_f = ConvBNReLU(channels, channels, k=(1, 1))  # context 1x1

        # Time path
        self.dw_t = ConvBNReLU(
            in_ch=channels,
            out_ch=channels,
            k=(1, k_t),
            p=(0, k_t // 2),
            groups=channels
        )
        self.pw_t  = ConvBNReLU(channels, channels, k=(1, 1))
        self.ctx_t = ConvBNReLU(channels, channels, k=(1, 1))  # context 1x1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frequency path
        yf = self.dw_f(x)                 # (B, C, F, T)
        yf = self.pw_f(yf)                # (B, C, F, T)
        vf = yf.mean(dim=2, keepdim=True) # (B, C, 1, T)
        vf = self.ctx_f(vf)               # (B, C, 1, T)
        yf = yf + vf                      # broadcast over F

        # Time path
        yt = self.dw_t(x)                 # (B, C, F, T)
        yt = self.pw_t(yt)                # (B, C, F, T)
        vt = yt.mean(dim=3, keepdim=True) # (B, C, F, 1)
        vt = self.ctx_t(vt)               # (B, C, F, 1)
        yt = yt + vt                      # broadcast over T

        # Sum back to C channels (same shape)
        out = self.sum_scale * (yf + yt)  # (B, C, F, T)
        return out
    

class TFParallelDWConvFullC(nn.Module):
    """
    Parallel TF block without pre-splitting channels.
    Each branch keeps C channels, then we concat (2C) and PW-compress to C.

    Input/Output: (B, C, F, T)
    """
    def __init__(self, channels: int, k_f: int = 3, k_t: int = 3):
        super().__init__()
        C = channels

        # Depthwise convs (axis-focused)
        self.dw_f = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(k_f, 1), padding=(k_f // 2, 0), groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),
        )
        self.dw_t = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=(1, k_t), padding=(0, k_t // 2), groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),
        )

        # Axis-wise context (global avg over the orthogonal axis), 1x1 gating on C
        # Frequency branch: avg over F -> shape (B,C,1,T)
        self.ctx_f = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=True),
            nn.Sigmoid(),  # gating
        )
        # Time branch: avg over T -> shape (B,C,F,1)
        self.ctx_t = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Fuse two C-channel branches -> (2C) then compress back to C
        self.fuse_pw = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, T = x.shape

        # Frequency-focused path (depthwise along F)
        yf = self.dw_f(x)                        # (B,C,F,T)
        af = yf.mean(dim=2, keepdim=True)        # (B,C,1,T) — avg over F
        wf = self.ctx_f(af)                      # (B,C,1,T) — gate in [0,1]
        yf = yf * (1.0 + wf)                     # multiplicative boost; use (yf + af) for additive

        # Time-focused path (depthwise along T)
        yt = self.dw_t(x)                        # (B,C,F,T)
        at = yt.mean(dim=3, keepdim=True)        # (B,C,F,1) — avg over T
        wt = self.ctx_t(at)                      # (B,C,F,1)
        yt = yt * (1.0 + wt)

        # Concat full-C branches, then 1x1 to mix and return to C
        out = torch.cat([yf, yt], dim=1)         # (B,2C,F,T)
        out = self.fuse_pw(out)                  # (B,C,F,T)

        return out


class TimeFrequencyAttention_by_Lin(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        fam_layers: int = 3,
        tam_layers: int = 3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.C = channels
        self.use_residual = use_residual
        hidden = max(1, channels // reduction)

        self._cam_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self._cam_avg = nn.AdaptiveAvgPool2d(1)
        self._cam_max = nn.AdaptiveMaxPool2d(1)
        self._sigmoid = nn.Sigmoid()

        self._fam = self._make_axis_stack(
            in_ch=2, mid_ch=2, out_ch=channels, kernel=(3, 1), padding=(1, 0), num_layers=fam_layers
        )
        self._tam = self._make_axis_stack(
            in_ch=2, mid_ch=2, out_ch=channels, kernel=(1, 3), padding=(0, 1), num_layers=tam_layers
        )

        self._fuse_1x1 = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)

    @staticmethod
    def _make_axis_stack(in_ch: int, mid_ch: int, out_ch: int, kernel, padding, num_layers: int):
        layers = []
        for i in range(max(1, num_layers) - 1):
            layers += [
                nn.Conv2d(in_ch if i == 0 else mid_ch, mid_ch, kernel_size=kernel, padding=padding, bias=True),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(mid_ch if num_layers > 1 else in_ch, out_ch, kernel_size=kernel, padding=padding, bias=True)]
        return nn.Sequential(*layers)

    def _cam(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self._cam_mlp(self._cam_avg(x))
        max_out = self._cam_mlp(self._cam_max(x))
        attn = self._sigmoid(avg_out + max_out)
        return x * attn

    def _fam_apply(self, fc: torch.Tensor) -> torch.Tensor:
        x_axis = fc.mean(dim=3, keepdim=True)             # (N,C,H,1)
        avg_c = x_axis.mean(dim=1, keepdim=True)          # (N,1,H,1)
        max_c, _ = x_axis.max(dim=1, keepdim=True)        # (N,1,H,1)
        feat = torch.cat([avg_c, max_c], dim=1)           # (N,2,H,1)
        attn_axis = self._sigmoid(self._fam(feat))        # (N,C,H,1)
        attn = attn_axis.expand(-1, -1, -1, fc.size(3))   # (N,C,H,W)
        return fc * attn

    def _tam_apply(self, fc: torch.Tensor) -> torch.Tensor:
        x_axis = fc.mean(dim=2, keepdim=True)             # (N,C,1,W)
        avg_c = x_axis.mean(dim=1, keepdim=True)          # (N,1,1,W)
        max_c, _ = x_axis.max(dim=1, keepdim=True)        # (N,1,1,W)
        feat = torch.cat([avg_c, max_c], dim=1)           # (N,2,1,W)
        attn_axis = self._sigmoid(self._tam(feat))        # (N,C,1,W)
        attn = attn_axis.expand(-1, -1, fc.size(2), -1)   # (N,C,H,W)
        return fc * attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc = self._cam(x)
        ff = self._fam_apply(fc)
        ft = self._tam_apply(fc)
        out = self._fuse_1x1(torch.cat([ff, ft], dim=1))
        return out + x if self.use_residual else out


class TimeFrequencyAttentionZhang(nn.Module):
    """
    TFA de Zhang en format NCHW:
      - Entrée  : x (N, C, H, W)  [H = fréquences (D), W = temps (L)]
      - Sortie  : x' (N, C, H, W) = x * (FA ⊗ TA)  (+ résiduel optionnel)
    Deux Conv1D dilatées (dilations 1 puis 2) sont utilisées dans chaque branche
    pour capturer du contexte long-terme le long de l'axe correspondant.

    Args:
        k_tfa (int): taille de noyau des Conv1D (impair recommandé: 3,5,7…)
        use_residual (bool): ajoute un skip (x' += x)
    """
    def __init__(self, k_tfa: int = 3, use_residual: bool = True):
        super().__init__()
        assert isinstance(k_tfa, int) and k_tfa >= 1
        self.k_tfa = k_tfa
        self.use_residual = use_residual

        pad1 = ((k_tfa - 1) // 2) * 1
        pad2 = ((k_tfa - 1) // 2) * 2

        # Branche TA (temps, longueur W) : (N*C, 1, W) -> (N*C, 1, W)
        self.ta_conv1 = nn.Conv1d(1, 1, kernel_size=k_tfa, dilation=1, padding=pad1, bias=True)
        self.ta_conv2 = nn.Conv1d(1, 1, kernel_size=k_tfa, dilation=2, padding=pad2, bias=True)

        # Branche FA (fréquence, longueur H) : (N*C, 1, H) -> (N*C, 1, H)
        self.fa_conv1 = nn.Conv1d(1, 1, kernel_size=k_tfa, dilation=1, padding=pad1, bias=True)
        self.fa_conv2 = nn.Conv1d(1, 1, kernel_size=k_tfa, dilation=2, padding=pad2, bias=True)

        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)  [H = fréquences, W = temps]
        return: (N, C, H, W)
        """
        N, C, H, W = x.shape
        # --- Agrégation globale par canal ---
        # TA: moyenne sur la fréquence -> (N, C, W)
        z_t = x.mean(dim=2)                                    # (N, C, W)
        # FA: moyenne sur le temps -> (N, C, H)
        z_f = x.mean(dim=3)                                    # (N, C, H)

        # --- Branche temporelle (TA) ---
        ta = z_t.reshape(N * C, 1, W)                          # (N*C, 1, W)
        ta = self.ta_conv1(ta)
        ta = self.act(ta)
        ta = self.ta_conv2(ta)
        ta = self.gate(ta).reshape(N, C, W)                    # (N, C, W)

        # --- Branche fréquentielle (FA) ---
        fa = z_f.reshape(N * C, 1, H)                          # (N*C, 1, H)
        fa = self.fa_conv1(fa)
        fa = self.act(fa)
        fa = self.fa_conv2(fa)
        fa = self.gate(fa).reshape(N, C, H)                    # (N, C, H)

        # --- Carte 2D par canal: (FA ⊗ TA) -> (N, C, H, W)
        #   FA_map: (N,C,H,1), TA_map: (N,C,1,W)
        fa_map = fa.unsqueeze(-1)                               # (N, C, H, 1)
        ta_map = ta.unsqueeze(-2)                               # (N, C, 1, W)
        tfa_map = fa_map * ta_map                               # (N, C, H, W)

        out = x * tfa_map
        if self.use_residual:
            out = out + x
        return out


class TFSepBlock(nn.Module):
    """
    Empile n unités (ch -> ch), puis applique un skip unique si residual=True.
    On suppose stride=1.

    mode:
      - 'classic'       -> TFSepConvs (implémentation existante)
      - 'parallel'      -> TFParallelDWConv (C -> C/2 + C/2 concat)
      - 'parallel_sum'  -> TFParallelDWSum (C -> C, somme des branches)
      - 'lin'/'Lin'     -> TimeFrequencyAttention_by_Lin (C -> C)   # ← NEW

    tfa_kwargs:
      - Dictionnaire passé à TimeFrequencyAttention_by_Lin
        (ex: dict(reduction=8, fam_layers=3, tam_layers=3, use_residual=True))
    """
    def __init__(self,
                 ch: int,
                 n: int = 1,
                 residual: bool = False,
                 k_f: int = 3,
                 k_t: int = 3,
                 shuffle_groups: int = 2,
                 mode: str = 'parallel',
                 sum_scale: float = 1.0,
                 tfa_kwargs: Optional[Dict] = None,
                 zhang_kwargs: Optional[Dict] = None):   # ← NEW
        super().__init__()
        assert n >= 1
        self.residual = residual

        if tfa_kwargs is None:
            tfa_kwargs = dict(reduction=8, fam_layers=3, tam_layers=3, use_residual=True)
        if zhang_kwargs is None:                          # ← NEW
            zhang_kwargs = dict(k_tfa=3, use_residual=True)

        layers = []
        for _ in range(n):
            m = mode.lower()
            if m == 'cai':
                layers.append(TFSepConvs(
                    in_channels=ch,
                    out_channels=ch,
                    k_f=k_f, k_t=k_t,
                    shuffle_groups=shuffle_groups
                ))
            elif m == 'parallel':
                layers.append(TFParallelDWConv(
                    channels=ch,
                    k_f=k_f, k_t=k_t
                ))
            elif m == 'parallel_sum':
                layers.append(TFParallelDWSum(
                    channels=ch,
                    k_f=k_f, k_t=k_t,
                    sum_scale=sum_scale
                ))
            elif m == 'parallel_fc':
                layers.append(TFParallelDWConvFullC(
                    channels=ch,
                     k_f=k_f, k_t=k_t
                ))
            elif m == 'lin':
                layers.append(TimeFrequencyAttention_by_Lin(
                    channels=ch, **tfa_kwargs
                ))
            elif m == 'zhang':
                layers.append(TimeFrequencyAttentionZhang(**zhang_kwargs))
            else:
                raise ValueError(f"Unknown mode '{mode}'")

        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.residual and y.shape == x.shape:
            y = y + x
        return y
