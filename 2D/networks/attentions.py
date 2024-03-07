import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import functools
import math
import timm
from timm.models.layers import DropPath, to_2tuple
import einops
from fvcore.nn import FlopCountAnalysis


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums


class GatedAttention(nn.Module):
    def __init__(self, in_chn, out_chn, dw_attn=False):
        super().__init__()

        if dw_attn == False:

            self.W_g = nn.Sequential(
                nn.Conv2d(in_channels=in_chn,
                          out_channels=out_chn,
                          kernel_size=1,
                          stride=1),
                nn.BatchNorm2d(num_features=out_chn)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(in_channels=in_chn,
                          out_channels=out_chn,
                          kernel_size=1,
                          stride=1),
                nn.BatchNorm2d(num_features=out_chn)
            )

            self.W = nn.Sequential(
                nn.Conv2d(in_channels=in_chn,
                          out_channels=1,
                          kernel_size=1,
                          stride=1),
                nn.BatchNorm2d(num_features=1),
                nn.Sigmoid()
            )  # Attention Score (B , 1 , H , W)

        else:

            self.W_g = nn.Sequential(
                nn.Conv2d(in_channels=in_chn,
                          out_channels=out_chn,
                          kernel_size=1,
                          stride=1,
                          groups=in_chn),
                nn.BatchNorm2d(num_features=out_chn)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(in_channels=in_chn,
                          out_channels=out_chn,
                          kernel_size=1,
                          stride=1,
                          groups=in_chn),
                nn.BatchNorm2d(num_features=out_chn)
            )

            self.W = nn.Sequential(
                nn.Conv2d(in_channels=in_chn,
                          out_channels=1,
                          kernel_size=1,
                          stride=1,
                          groups=1),
                nn.BatchNorm2d(num_features=1),
                nn.Sigmoid()
            )  # Attention Score (B , 1 , H , W)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        w_x = self.W_x(x)
        g_x = self.W_g(g)

        aggr = w_x + g_x

        aggr = self.relu(aggr)

        attn_score = self.W(aggr)

        return attn_score


class SelectiveKernelAttn(nn.Module):
    """
    Selective Kernel Convolution Module

    As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications (DWConv and DW-D Conv).

    The code is developed by Sina Ghorbani aka (Colorless Tsukoro Tazaki)
    Github: https://github.com/sinagho

    Input: torch.tensor (BxCxHxW)
    Output: torch.tensor (BxCxHxW)
    Args:
        channels (int): module input/output (feature) channel count
        kernel_size (int): kernel size for each convolution branch (PS: Kernel size 3 is suitable as paper said)
        dilation (int): Dilation rate for increasing receptive field of each convolution with their kernel size (kernel_size = 3)
        reduction (int): Reduction ratio for reducing complexity
        l (int): Minimal Value
        num_path: total number of convolutinal path (this code is just based on 2 path selective kernel)

    """

    def __init__(self, channels, kernel_size=3, dilation=2, reduction=2, min_val=32, num_path=2):
        super().__init__()

        padding = 1
        max_d = max((channels // reduction, min_val))
        self.num_path = num_path

        self.path_one = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=kernel_size,
                      stride=1,
                      groups=channels,
                      padding=padding),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True)
        )

        self.path_two = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=kernel_size,
                      stride=1,
                      groups=channels,  # Conv2d --> DWConv
                      padding=padding + 1,  # padding + 1 is cruicial to preserve spatial resolution
                      dilation=dilation),
            # dilation is used to transform the DWConv to DW-Dconv ---> receptive field is equal to (5,5)
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True)
        )

        self.global_descriptor = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.reducer = nn.Conv2d(in_channels=channels,
                                 out_channels=max_d,
                                 kernel_size=1)

        self.selection = nn.Conv2d(min_val,
                                   channels * num_path,
                                   kernel_size=1,
                                   bias=False)  # A and B matrix Builder (Both of them are learnable)

    def forward(self, x):
        # Split
        u_tilde = self.path_one(x)
        u_hat = self.path_two(x)

        stacked_u = torch.stack((u_tilde, u_hat),
                                dim=1)  # for final prod (weights * stacked_feats) # B x paths x C x H x W
        # Fuse

        u = u_tilde + u_hat  # B x C x H x W
        u = self.global_descriptor(u)  # B x C x 1 x 1

        z = F.relu(self.reducer(u))  # B x (max_d) x 1 x 1

        # Select
        select = self.selection(z)  # B x (max_d * C ) x 1 x 1 , max_d * C  = 128 if C is 64
        B, C, H, W = select.shape
        num_path = self.num_path

        select_ = select.view(B, num_path, C // num_path, H, W)  # B x num_path x (128) // 2 x 1 x 1
        weights = torch.softmax(select_, dim=1)  # B x num_path X C x 1 x 1

        prod = weights * stacked_u  # B x num_path x C x H x W

        v = torch.sum(prod, dim=1)  # B x C x H x W

        return v


class UpSampling(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        dim_in = dim_out * 2

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim_in, out_channels=dim_out,
                               kernel_size=1),
            nn.BatchNorm2d(num_features=dim_out),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

    def forward(self, x):
        y = self.block(x)
        return y

class SCLKattention(nn.Module):
  """
  Shared-weights Context-aware Large Kernel Attention (SCLKA)

  SCLKA attention is a modified version of Large kerenl attention which captures Long-range dependecies,
  Local-interactions in Shared weight and Context-aware Design simultaneously while it is known as linear attetnion as well.

  The Attention mechanism and the code are developed by Sina Ghorbani A.K.A (Colorless Tsukoro Tazaki)
  Github: https://github.com/sinagho

  Input: torch.tensor (BxCxHxW)
  Output: torch.tensor (BxCxHxW)
  Args:
      dim: (int): input/output dimension
  """
  def __init__(self, dim):
    super().__init__()
    # Context-Aware
    self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
    self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

    #self.conv1 = nn.Conv2d(dim, dim, 1)

    # Shared
    self.conv_shared = nn.Conv2d(dim, dim, 5, padding = 2, groups = dim)

  def forward(self, x):
    x_ = x.clone()
    x_s = self.conv_shared(x)
    x = self.conv0(x)
    x = self.conv_spatial(x)

    #x = self.conv1(x)

    attn_map = x_s * x

    return x_ + attn_map

class InvertedDualBranchLKA(nn.Module):
    """
    Inverted Multi Branch LKA is developed based on LKA and SCLKA to enhance the feature extraction performance of SCLKA (LKA) built upon Inverted Residual Block.
    It has two inputs, One of them is directed from the encoder current encoder and the other on is the semantic constructed feature map from the next encoder layer to
    represent the low frequency features as high quality as possible (semantic details are cruicial to generate low frequency features).
    Finally, by using this plug-in module instead of simple dilated convolution in Selective Kernel Network's second branch (low freq),
    low frequnecy (global) features can be enhanced


    The Module and the code are developed by Sina Ghorbani A.K.A (Colorless Tsukoro Tazaki)
    Github: https://github.com/sinagho

    Args:
        dim: (int): input/output dimension
        expansion_rate: (int):  expansion rate for dimensions (Inverted Block)
    """

    def __init__(self, dim, expansion_rate=4):
        super().__init__()
        self.expansion_rate = expansion_rate
        self.dim_in = dim * 2  # 324 x 2 = 728
        # Branch One
        self.proj_1_1 = self.projection_1x1_1(dim)
        self.attn_1 = SCLKattention(dim * self.expansion_rate)
        self.proj_1_2 = self.projection_1x1_2(dim)

        # Branch Two
        self.ups = UpSampling(self.dim_in, dim)
        self.proj_2_1 = self.projection_1x1_1(dim)
        self.attn_2 = SCLKattention(dim * self.expansion_rate)
        self.proj_2_2 = self.projection_1x1_2(dim)

        # Fuse
        self.fuse = self.final_proj(dim)

    def projection_1x1_1(self, dim):
        proj_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim * self.expansion_rate, 1, 1),
            nn.BatchNorm2d(num_features=dim * self.expansion_rate),
            nn.ReLU(inplace=True)
        )
        return proj_1x1

    def projection_1x1_2(self, dim):
        proj_1x1 = nn.Sequential(
            nn.Conv2d(dim * self.expansion_rate, dim, 1, 1),
            nn.BatchNorm2d(num_features=dim),
        )
        return proj_1x1

    def final_proj(self, dim):
        proj = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.BatchNorm2d(num_features=dim),
            # nn.GELU()
        )
        return proj

    def forward(self, x_0, x_1):
        x_0_ = x_0.clone()
        x_1 = self.ups(x_1)  # Upsampled x_1
        x_1_ = x_1.clone()
        print(x_1_.shape)

        # Branch One (encoder layer: l)
        x_0 = self.proj_1_1(x_0)
        x_0 = self.attn_1(x_0)
        x_0 = self.proj_1_2(x_0)

        x_0 = x_0 + x_0_

        # Branch Two (encoder layer: l + 1)
        x_1 = self.proj_2_1(x_1)
        x_1 = self.attn_2(x_1)
        x_1 = self.proj_2_2(x_1)

        x_1 = x_1 + x_1_

        # Fuse
        x = torch.cat((x_0, x_1), dim=1)
        x = self.fuse(x)

        return x


class SelectiveKernelAttn_V2(nn.Module):
  """
  Selective Kernel Convolution Module

  As described in Selective Kernel Networks (https://arxiv.org/abs/1903.06586) with some modifications (DWConv and DW-D Conv).

  The code is developed by Sina Ghorbani aka (Colorless Tsukoro Tazaki)
  Github: https://github.com/sinagho

  Input: torch.tensor (BxCxHxW)
  Output: torch.tensor (BxCxHxW)
  Args:
      channels (int): module input/output (feature) channel count
      kernel_size_local (int): kernel size for each convolution branch (PS: Kernel size 3 is suitable as paper said)
      dilation (int): Dilation rate for increasing receptive field of each convolution with their kernel size (kernel_size = 3)
      reduction (int): Reduction ratio for reducing complexity
      l (int): Minimal Value
      num_path (int): total number of convolutinal path (this code is just based on 2 path selective kernel)
      num_descriptor (int): enhance the global feature descriptor. 1: (avg), 2: (max + avg), 3: (max + avg + std) [Style + Context]

  """
  def __init__(self, channels, kernel_size_local = 3, dilation = 2, reduction = 2, l = 32, num_path = 2, num_descriptor = 2):
    super().__init__()

    padding = 1
    max_d = max((channels // reduction , l ))
    self.num_path = num_path
    self.num_descriptor = num_descriptor

    self.path_one = nn.Sequential(
        nn.Conv2d(in_channels= channels,
                  out_channels= channels,
                  kernel_size = kernel_size_local,
                  stride = 1,
                  groups = channels,
                  padding = padding),
        nn.BatchNorm2d(num_features= channels),
        nn.ReLU(inplace = True)
    )

    self.path_two = InvertedDualBranchLKA(dim = channels, expansion_rate = 4)

    # self.path_two = nn.Sequential(
    #     nn.Conv2d(in_channels= channels,
    #               out_channels= channels,
    #               kernel_size = kernel_size_local,
    #               stride = 1,
    #               groups = channels, # Conv2d --> DWConv
    #               padding = padding + 1, # padding + 1 is cruicial to preserve spatial resolution
    #               dilation = dilation), # dilation is used to transform the DWConv to DW-Dconv ---> receptive field is equal to (5,5)
    #     nn.BatchNorm2d(num_features= channels),
    #     nn.ReLU(inplace = True)
    # )
    if num_descriptor == 1:

      self.global_descriptor_avg = nn.AdaptiveAvgPool2d(output_size = (1,1))
    elif num_descriptor == 2:
      self.global_descriptor_avg = nn.AdaptiveAvgPool2d(output_size = (1,1))
      self.global_descriptor_max = nn.AdaptiveMaxPool2d(output_size=(1,1))
      self.mixer = nn.Conv2d(in_channels = channels * num_descriptor,
                             out_channels= channels,
                             kernel_size= 1)
    elif num_descriptor == 3:
      self.global_descriptor_avg = nn.AdaptiveAvgPool2d(output_size = (1,1))
      self.global_descriptor_max = nn.AdaptiveMaxPool2d(output_size=(1,1))
      #self.global_descriptor_std = self.AdaptiveStdPool2d
      self.mixer = nn.Conv2d(in_channels = channels * num_descriptor,
                             out_channels= channels,
                             kernel_size= 1)


    self.reducer = nn.Conv2d(in_channels = channels,
                             out_channels = max_d,
                             kernel_size = 1)

    #self.selection = nn.Conv2d(max_d, channels * num_path, kernel_size=1, bias=False) # A and B matrix Builder (Both of them are learnable)
    self.selection = nn.Conv2d(channels, channels * num_path, kernel_size=1, bias=False)
  def AdaptiveStdPool2d(self,x):
    b , c , h , w = x.shape # B, C, H, W
    x = x.view(b, c, -1).std(dim = -1, keepdim = True).unsqueeze(-1).contiguous() # B, C, H, W --> B, C, H x W --> B, C, 1 --> B, C, 1, 1
    return x

  def forward(self,x, y):

    # Split
    u_tilde = self.path_one(x)
    u_hat = self.path_two(x, y)

    stacked_u = torch.stack((u_tilde, u_hat), dim = 1) # for final prod (weights * stacked_feats) # B x paths x C x H x W
    # Fuse

    u_0 = u_tilde + u_hat # B x C x H x W

    u = self.global_descriptor_avg(u_0) # B x C x 1 x 1

    if self.num_descriptor == 2:
      u_max = self.global_descriptor_max(u_0) # B x C x 1 x 1

      u = torch.cat([u, u_max], dim=1) # B x 2C x 1 x 1

      u = self.mixer(u) # B x C x 1 x 1

    if self.num_descriptor == 3:
      u_max = self.global_descriptor_max(u_0) # B x C x 1 x 1
      u_std = self.AdaptiveStdPool2d(u_0) # B x C x 1 x 1

      u = torch.cat([u, u_max, u_std], dim=1) # B x 3C x 1 x 1

      u = self.mixer(u) # B x C x 1 x 1

    z = F.relu(self.reducer(u)) #  B x (max_d) x 1 x 1
    z = F.relu(u)
    # Select

    select = self.selection(z) # B x (max_d * C ) x 1 x 1 , max_d * C  = 128
    B, C, H, W = select.shape
    num_path = self.num_path


    select_ = select.view(B, num_path, C // num_path, H, W) # B x num_path x (128) // 2 x 1 x 1 (A and B)
    weights = torch.softmax(select_, dim=1) # B x num_path X C x 1 x 1

    prod = weights * stacked_u # B x num_path x C x H x W


    v = torch.sum(prod, dim = 1) # B x C x H x W

    return v

class GlobalExtraction(nn.Module):
  def __init__(self,dim = None):
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, 1, 1,1),
        nn.BatchNorm2d(1)
    )
  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj

class ContextExtraction(nn.Module):
  def __init__(self, dim, reduction = None):
    super().__init__()
    self.reduction = 1 if reduction == None else 2

    self.dconv = self.DepthWiseConv2dx2(dim)
    self.proj = self.Proj(dim)

  def DepthWiseConv2dx2(self, dim):
    dconv = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 1,
              groups = dim),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True),
        nn.Conv2d(in_channels = dim,
              out_channels = dim,
              kernel_size = 3,
              padding = 2,
              dilation = 2),
        nn.BatchNorm2d(num_features = dim),
        nn.ReLU(inplace = True)
    )
    return dconv

  def Proj(self, dim):
    proj = nn.Sequential(
        nn.Conv2d(in_channels = dim,
              out_channels = dim //self.reduction,
              kernel_size = 1
              ),
        nn.BatchNorm2d(num_features = dim//self.reduction)
    )
    return proj
  def forward(self,x):
    x = self.dconv(x)
    x = self.proj(x)
    return x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= ContextExtraction(dim)
    self.global_ = GlobalExtraction()
    self.bn = nn.BatchNorm2d(num_features=dim)

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse


class MultiScaleGatedAttn(nn.Module):
    # Version 1
    # attention map channel is C
    # Softmax is chosen
    # residual
    # interaction addition is multiplation
    # batch norm
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)
    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels= dim, out_channels= dim,
                  kernel_size= 3, padding= 1, stride= 1),
        nn.BatchNorm2d(num_features= dim),
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1),
        nn.BatchNorm2d(num_features=dim)
    )

  def forward(self,x,g):
    x_ = x.clone()
    g_ = g.clone()

    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

    multi = self.multi(x, g) # B, C, H, W

    ### Option 1 ###
    # b,c,h,w = multi.size()


    # score = multi.view(b,c,-1).softmax(dim = -1)

    # attention_maps = score.view(b, c, h, w)
    # x_attention = attention_maps * x_
    # g_attention = (1 - attention_maps) * g_

    ### Option 1 ###

    ### Option 2 ###
    multi = self.selection(multi) # B, num_path, H, W

    attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
    #attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_

    x_att = x_att + x_
    g_att = g_att + g_
    ## Bidirectional Interaction

    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att


    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att

    interaction = x_att_2 * g_att_2

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    y = self.bn_2(weighted + y)
    return y

class MultiScaleGatedAttn_soft_1_res(nn.Module):
    # Version 1_plus
    # attention map channel is 1
    # Softmax is chosen
    # residual
    # interaction addition is multiplation
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2,1)
        self.proj = nn.Conv2d(dim, 1,1)
        self.bn = nn.BatchNorm2d(1)
        self.bn_2 = nn.BatchNorm2d(1)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=dim)
        )
    def forward(self,x,g):

        x_ = x.clone()
        g_ = g.clone()

        #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

        multi = self.multi(x, g) # B, C, H, W

        ### Option 1 ###
        # b,c,h,w = multi.size()


        # score = multi.view(b,c,-1).softmax(dim = -1)

        # attention_maps = score.view(b, c, h, w)
        # x_attention = attention_maps * x_
        # g_attention = (1 - attention_maps) * g_

        ### Option 1 ###

        ### Option 2 ###
        multi = self.selection(multi) # B, num_path, H, W

        attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
        #attention_weights = torch.sigmoid(multi)
        A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

        x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
        g_att = B.expand_as(g_) * g_

        x_att = x_att + x_
        g_att = g_att + g_
        ## Bidirectional Interaction

        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att


        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att

        interaction = x_att_2 * g_att_2

        projected = torch.sigmoid(self.bn(self.proj(interaction)))

        weighted = projected * x_

        y = self.conv_block(weighted)

        y = self.bn_2(weighted + y)

        return y


class MultiScaleGatedAttnV2(nn.Module):
    # Version 2
    # attention map channel is C
    # Sigmoid is chosen
    # residual
    # interaction addition is multiplation
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=dim)
        )

    def forward(self, x, g):
        x_ = x.clone()
        g_ = g.clone()

        # stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

        multi = self.multi(x, g)  # B, C, H, W

        ### Option 1 ###
        # b,c,h,w = multi.size()

        # score = multi.view(b,c,-1).softmax(dim = -1)

        # attention_maps = score.view(b, c, h, w)
        # x_attention = attention_maps * x_
        # g_attention = (1 - attention_maps) * g_

        ### Option 1 ###

        ### Option 2 ###
        multi = self.selection(multi)  # B, num_path, H, W

        # attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
        attention_weights = torch.sigmoid(multi)
        A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

        x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
        g_att = B.expand_as(g_) * g_

        x_att = x_att + x_
        g_att = g_att + g_
        ## Bidirectional Interaction

        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att

        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att

        interaction = x_att_2 * g_att_2

        projected = torch.sigmoid(self.bn(self.proj(interaction)))

        weighted = projected * x_

        y = self.conv_block(weighted)

        y = self.bn_2(weighted + y)

        return y

class MultiScaleGatedAttnV3_sum(nn.Module):
    # Version 3
    # attention map channel is C
    # Sigmoid is chosen
    # residual
    # interaction addition is summation
  def __init__(self, dim):
    super().__init__()
    self.multi = MultiscaleFusion(dim)
    self.selection = nn.Conv2d(dim, 2,1)
    self.proj = nn.Conv2d(dim, dim,1)
    self.bn = nn.BatchNorm2d(dim)
    self.bn_2 = nn.BatchNorm2d(dim)

    self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels= dim, out_channels= dim,
                  kernel_size= 3, padding= 1, stride= 1),
        nn.BatchNorm2d(num_features= dim),
        nn.Conv2d(in_channels=dim, out_channels=dim,
                  kernel_size=1, stride=1),
        nn.BatchNorm2d(num_features=dim))
  def forward(self,x, g):
    x_ = x.clone()
    g_ = g.clone()

    #stacked = torch.stack((x_, g_), dim = 1) # B, 2, C, H, W

    multi = self.multi(x, g) # B, C, H, W

    ### Option 1 ###
    # b,c,h,w = multi.size()


    # score = multi.view(b,c,-1).softmax(dim = -1)

    # attention_maps = score.view(b, c, h, w)
    # x_attention = attention_maps * x_
    # g_attention = (1 - attention_maps) * g_

    ### Option 1 ###

    ### Option 2 ###
    multi = self.selection(multi) # B, num_path, H, W

    #attention_weights = F.softmax(multi, dim=1)  # Shape: [B, 2, H, W]
    attention_weights = torch.sigmoid(multi)
    A, B = attention_weights.split(1, dim=1)  # Each will have shape [B, 1, H, W]

    x_att = A.expand_as(x_) * x_  # Using expand_as to match the channel dimensions
    g_att = B.expand_as(g_) * g_

    x_att = x_att + x_
    g_att = g_att + g_
    ## Bidirectional Interaction

    x_sig = torch.sigmoid(x_att)
    g_att_2 = x_sig * g_att


    g_sig = torch.sigmoid(g_att)
    x_att_2 = g_sig * x_att

    interaction = x_att_2 + g_att_2 # Sum instead of multipliation

    projected = torch.sigmoid(self.bn(self.proj(interaction)))

    weighted = projected * x_

    y = self.conv_block(weighted)

    y = self.bn_2(weighted + y)

    return y
if __name__ == "__main__":
    xi = torch.randn(1, 192, 28, 28).cuda()
    #xi_1 = torch.randn(1, 384, 14, 14)
    g = torch.randn(1, 192, 28, 28).cuda()
    #ff = ContextBridge(dim=192)

    attn = MultiScaleGatedAttn_soft_1_res(dim = xi.shape[1]).cuda()

    print(attn(xi, g).shape)
