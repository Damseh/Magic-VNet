import warnings
import torch
import torch.nn as nn

from .blocks import *

__all__ = ('VNet', 'VNet_BSC', 'VNet_CSE', 'VNet_SSE', 'VNet_SCSE',
           'VNet_ASPP', 'VNet_MABN',
           'VBNet', 'VBNet_CSE', 'VBNet_SSE', 'VBNet_SCSE',
           'VBNet_ASPP',
           'SKVNet', 'SKVNet_ASPP')


class VNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNetBase, self).__init__()
        norm_type = nn.BatchNorm3d
        act_type = nn.ReLU
        se_type = None
        drop_type = None
        feats = [16, 32, 64, 128, 256]
        num_blocks = [1, 2, 3, 3]
        block_name = 'residual'
        self._use_aspp = False
        if 'norm_type' in kwargs.keys():
            norm_type = kwargs['norm_type']
        if 'num_blocks' in kwargs.keys():
            num_blocks = kwargs['num_blocks']            
        if 'act_type' in kwargs.keys():
            act_type = kwargs['act_type']
        if 'feats' in kwargs.keys():
            feats = kwargs['feats']
        if 'se_type' in kwargs.keys():
            se_type = kwargs['se_type']
        if 'num_blocks' in kwargs.keys():
            num_blocks = kwargs['num_blocks']
        if 'drop_type' in kwargs.keys():
            drop_type = kwargs['drop_type']
        if 'use_aspp' in kwargs.keys():
            self._use_aspp = kwargs['use_aspp']
        if 'block_name' in kwargs.keys():
            block_name = kwargs['block_name']

        self.in_conv = InputBlock(in_channels, feats[0],
                                  norm_type=norm_type,
                                  act_type=act_type)

        self.down1 = DownBlock(feats[0], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)
        self.down2 = DownBlock(feats[1], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.down3 = DownBlock(feats[2], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.down4 = DownBlock(feats[3], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                               drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        if self._use_aspp:
            self.aspp = ASPP(feats[4], dilations=[1, 2, 3, 4], norm_type=norm_type, act_type=act_type,
                             drop_type=drop_type)
        self.up4 = UpBlock(feats[4], feats[3], feats[4], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[3], block_name=block_name)
        self.up3 = UpBlock(feats[4], feats[2], feats[3], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[2], block_name=block_name)
        self.up2 = UpBlock(feats[3], feats[1], feats[2], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[1], block_name=block_name)
        self.up1 = UpBlock(feats[2], feats[0], feats[1], norm_type=norm_type, act_type=act_type, se_type=se_type,
                           drop_type=drop_type, num_blocks=num_blocks[0], block_name=block_name)

        self.out_block = OutBlock(feats[1], out_channels, norm_type, act_type)

        init_weights(self)

    def forward(self, input):
        if input.size(2) // 16 == 0 or input.size(3) // 16 == 0 or input.size(4) // 16 == 0:
            raise RuntimeError("input tensor shape is too small")
        input = self.in_conv(input)
        down1 = self.down1(input)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        if self._use_aspp:
            down4 = self.aspp(down4)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, input)

        out = self.out_block(up1)
        return out


class VNet_BSC(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_BSC, self).__init__(in_channels, out_channels, **kwargs)


class VNet_MABN(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_MABN, self).__init__(in_channels, out_channels, norm_type=MABN3d, **kwargs)


class VNet_CSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_CSE, self).__init__(in_channels, out_channels, se_type='cse', **kwargs)


class VNet_SSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_SSE, self).__init__(in_channels, out_channels, se_type='sse', **kwargs)


class VNet_SCSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_SCSE, self).__init__(in_channels, out_channels, se_type='scse', **kwargs)


class VNet_ASPP(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VNet_ASPP, self).__init__(in_channels, out_channels, use_aspp=True, **kwargs)


class VBNet(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet, self).__init__(in_channels, out_channels, block_name='bottleneck', **kwargs)


class VBNet_MABN(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_MABN, self).__init__(in_channels, out_channels,
                                         block_name='bottleneck',
                                         norm_type=MABN3d, **kwargs)


class VBNet_CSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_CSE, self).__init__(in_channels, out_channels,
                                        block_name='bottleneck',
                                        se_type='cse', **kwargs)


class VBNet_SSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_SSE, self).__init__(in_channels, out_channels,
                                        block_name='bottleneck',
                                        se_type='sse', **kwargs)


class VBNet_SCSE(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_SCSE, self).__init__(in_channels, out_channels,
                                         block_name='bottleneck',
                                         se_type='scse', **kwargs)


class VBNet_ASPP(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(VBNet_ASPP, self).__init__(in_channels, out_channels,
                                         block_name='bottleneck',
                                         use_aspp=True, **kwargs)


class SKVNet(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        if 'se_type' in kwargs.keys():
            warnings.warn('`se_type` keyword not working in `SKVNet`', UserWarning)
        super(SKVNet, self).__init__(in_channels, out_channels, block_name='sk', **kwargs)


class SKVNet_ASPP(VNetBase):
    def __init__(self, in_channels, out_channels, **kwargs):
        if 'se_type' in kwargs.keys():
            warnings.warn('`se_type` keyword not working in `SKVNet_ASPP`', UserWarning)
        super(SKVNet_ASPP, self).__init__(in_channels, out_channels, block_name='sk', use_aspp=True, **kwargs)





#-------------------------------------------------------#
# from https://raw.githubusercontent.com/mattmacy/vnet.pytorch/master/vnet.py
#-------------------------------------------------------#


import torch.nn.functional as F

def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
# class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'
#                              .format(input.dim()))
#         super(ContBatchNorm3d, self)._check_input_dim(input)

#     def forward(self, input):
#         self._check_input_dim(input)
#         return F.batch_norm(
#             input, self.running_mean, self.running_var, self.weight, self.bias,
#             True, self.momentum, self.eps)
    
ContBatchNorm3d=nn.BatchNorm3d

class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(16)
        self.relu1 = ELUCons(elu, 16)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 2)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False, **kwargs):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu, nll)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out









