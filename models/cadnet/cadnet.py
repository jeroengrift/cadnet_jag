import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import timm
import warnings
import torchvision

from collections import OrderedDict
from einops import rearrange
from timm.models.layers import DropPath
from models.basemodel import BaseModel, FocalLoss
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer

from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_, trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList, load_state_dict)
from mmcv.utils import to_2tuple
from mmseg.models.backbones.swin import SwinBlockSequence
from mmseg.utils import get_root_logger
from mmseg.models.utils.embed import PatchEmbed, PatchMerging
from mmcv.cnn.bricks.norm import build_norm_layer as mmcv_build_norm_layer
from mmcv.cnn.bricks.activation import build_activation_layer
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'

# UNetResNet-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            '''
            Parameters for Deconvolution were chosen to avoid artifacts, following
            link https://distill.pub/2016/deconv-checkerboard/
            '''

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(middle_channels),
                nn.ELU(),
                nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ELU()
            )

    def forward(self, x):
        return self.block(x)

def cat_non_matching(x1, x2):
    diffY = x1.size()[2] - x2.size()[2]
    diffX = x1.size()[3] - x2.size()[3]

    x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

    # for padding issues, see
    # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
    # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

    x = torch.cat([x1, x2], dim=1)
    return x

# COMES FROM FFL PAPER: https://github.com/Lydorn/Polygonization-by-Frame-Field-Learning
class UNetResNetBackbone(nn.Module):
    '''PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    '''

    def __init__(self, encoder_depth, num_filters=32, dropout_2d=0.2,
                 pretrained=False, is_deconv=False):
        super().__init__()
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of ResNet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)

        dec5 = self.dec5(cat_non_matching(conv5, center))
        dec4 = self.dec4(cat_non_matching(conv4, dec5))
        dec3 = self.dec3(cat_non_matching(conv3, dec4))
        dec2 = self.dec2(cat_non_matching(conv2, dec3))
        dec1 = self.dec1(dec2)

        y = F.dropout2d(dec1, p=self.dropout_2d)

        result = OrderedDict()
        result['out'] = y

        return result
    

# UNet -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class UNetBackbone(nn.Module):
    def __init__(self, n_channels, n_hidden_base, no_padding=False):
        super(UNetBackbone, self).__init__()
        self.no_padding = no_padding
        self.inc = InConv(n_channels, n_hidden_base, no_padding)
        self.down1 = Down(n_hidden_base, n_hidden_base*2, no_padding)
        self.down2 = Down(n_hidden_base*2, n_hidden_base*4, no_padding)
        self.down3 = Down(n_hidden_base*4, n_hidden_base*8, no_padding)
        self.down4 = Down(n_hidden_base*8, n_hidden_base*16, no_padding)

        self.up1 = Up(n_hidden_base*16, n_hidden_base*8, n_hidden_base*8, no_padding)
        self.up2 = Up(n_hidden_base*8, n_hidden_base*4, n_hidden_base*4, no_padding)
        self.up3 = Up(n_hidden_base*4, n_hidden_base*2, n_hidden_base*2, no_padding)
        self.up4 = Up(n_hidden_base*2, n_hidden_base, n_hidden_base, no_padding)

    def forward(self, x):
        x0 = self.inc.forward(x)
        x1 = self.down1.forward(x0)
        x2 = self.down2.forward(x1)
        x3 = self.down3.forward(x2)
        
        y4 = self.down4.forward(x3)
        y3 = self.up1.forward(y4, x3)
        y2 = self.up2.forward(y3, x2)
        y1 = self.up3.forward(y2, x1)
        y0 = self.up4.forward(y1, x0)

        result = OrderedDict()
        result['out'] = y0
    
        return result
    

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, no_padding):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0 if no_padding else 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=0 if no_padding else 1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, no_padding):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, no_padding)

    def forward(self, x):
        x = self.conv.forward(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, no_padding):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, no_padding)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, no_padding):
        super(Up, self).__init__()
        self.conv = DoubleConv(in_ch_1 + in_ch_2, out_ch, no_padding)

    def forward(self, x1, x2):
        # x1 torch.Size([2, 1024, 32, 32])
        # x2 torch.Size([2, 512, 64, 64])
        
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=False)
        # x1 torch.Size([2, 1024, 64, 64])

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            
        x = torch.cat([x2, x1], dim=1)  
        # x torch.Size([2, 1536, 64, 64])      
        x = self.conv.forward(x)
        # x torch.Size([2, 512, 64, 64])

        return x


# SWIN --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# https://github.com/UARK-AICV/AerialFormer/blob/e09134d685e17f30deba7b13b2b416e81f6f1375/aerialseg/models/backbones/swin_stem.py#L73
class SwinStemTransformer(BaseModule):
    def __init__(self,
                 pretrain_img_size=384,
                 in_channels=3,
                 embed_dims=128,
                 patch_size=4,
                 window_size=12,
                 mlp_ratio=4,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', requires_grad=True),
                 conv_norm_cfg=dict(type='SyncBN', requires_grad=True),
                 with_cp=False,
                 pretrained=None,
                 frozen_stages=-1,
                 init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth')
                 ):
        
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn("Deprecation Warning: pretrained is deprecated, please use init_cfg instead")
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinStemTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        inplanes = 64
        self.stem = nn.Sequential(
            ConvModule(in_channels, inplanes, kernel_size=3, stride=2, padding=1, norm_cfg=conv_norm_cfg, act_cfg=act_cfg),
            ConvModule(inplanes, inplanes, kernel_size=3, stride=1, padding=1, norm_cfg=conv_norm_cfg, act_cfg=act_cfg),
            ConvModule(inplanes, inplanes, kernel_size=3, stride=1, padding=1, norm_cfg=conv_norm_cfg, act_cfg=act_cfg),
            ConvModule(inplanes, int(embed_dims/2), kernel_size=1, stride=1, norm_cfg=conv_norm_cfg, act_cfg=act_cfg)
        )

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))
        self.embed_dims = embed_dims
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        '''Convert the model into training mode while keep layers freezed.'''
        super(SwinStemTransformer, self).train(mode)
        self._freeze_stages()

    # for finetuning: https://medium.com/@hassaanidrees7/fine-tuning-transformers-techniques-for-improving-model-performance-4b4353e8ba93
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        conv_x = self.stem(x)
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = [conv_x]
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
                
        return outs


class SwinUnetDecoder(nn.Module):
    def __init__(self, dropout_2d=0.2, is_deconv=False):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.pool = nn.MaxPool2d(2, 2)
        self.center = DecoderBlockV2(1024, 512, 256, is_deconv)
        self.dec5 = DecoderBlockV2(1280, 512, 256, is_deconv)
        self.dec4 = DecoderBlockV2(768, 512, 256, is_deconv)
        self.dec3 = DecoderBlockV2(512, 128, 64, is_deconv)
        self.dec2 = DecoderBlockV2(192, 128, 128, is_deconv)
        self.dec1 = DecoderBlockV2(128, 128, 32, is_deconv)
        
    def forward(self, inputs):
        conv5 = inputs[4]
        conv4 = inputs[3]
        conv3 = inputs[2]
        conv2 = inputs[1]
        
        # # do not use stem
        # conv1 = inputs[0]
        
        pool = self.pool(conv5)        
        center = self.center(pool)
        dec5 = self.dec5(cat_non_matching(conv5, center))
        dec4 = self.dec4(cat_non_matching(conv4, dec5))
        dec3 = self.dec3(cat_non_matching(conv3, dec4))
        dec2 = self.dec2(cat_non_matching(conv2, dec3))
        dec1 = self.dec1(dec2)

        y = F.dropout2d(dec1, p=self.dropout_2d)
        
        result = OrderedDict()
        result['out'] = y

        return result


class SwinUnetMultiCoaDecoder(nn.Module):
    def __init__(self, dropout_2d=0.2, is_deconv=False):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.pool = nn.MaxPool2d(2, 2)
        self.center = DecoderBlockV2(1024, 512, 256, is_deconv)
        self.dec5 = DecoderBlockV2(1280, 512, 256, is_deconv)
        self.dec4 = DecoderBlockV2(785, 512, 256, is_deconv)
        self.dec3 = DecoderBlockV2(529, 256, 64, is_deconv)
        self.dec2 = DecoderBlockV2(209, 128, 128, is_deconv)
        self.dec1 = DecoderBlockV2(209, 128, 32, is_deconv)
        
        self.stage_seg_5 = nn.Sequential(
            torch.nn.Conv2d(1280, 1280, 3, padding=1),
            torch.nn.BatchNorm2d(1280),
            torch.nn.ELU(),
            torch.nn.Conv2d(1280, 1, 1))
        
        self.stage_coa_5_d1 = nn.Sequential(
            nn.Conv2d(1280, 1280, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1280, 8, 3, padding=1, dilation=1),
            SELayer(8))
        
        self.stage_coa_5_d3 = nn.Sequential(
            nn.Conv2d(1280, 1280, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1280, 8, 3, padding=3, dilation=3),
            SELayer(8))
        
        self.stage_seg_4 = nn.Sequential(
            torch.nn.Conv2d(785, 785, 3, padding=1),
            torch.nn.BatchNorm2d(785),
            torch.nn.ELU(),
            torch.nn.Conv2d(785, 1, 1))
        
        self.stage_coa_4_d1 = nn.Sequential(
            nn.Conv2d(785, 785, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(785, 8, 3, padding=1, dilation=1),
            SELayer(8))
        
        self.stage_coa_4_d3 = nn.Sequential(
            nn.Conv2d(785, 785, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(785, 8, 3, padding=3, dilation=3),
            SELayer(8))
    
        self.stage_seg_3 = nn.Sequential(
            torch.nn.Conv2d(529, 529, 3, padding=1),
            torch.nn.BatchNorm2d(529),
            torch.nn.ELU(),
            torch.nn.Conv2d(529, 1, 1))
        
        self.stage_coa_3_d1 = nn.Sequential(
            nn.Conv2d(529, 529, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(529, 8, 3, padding=1, dilation=1),
            SELayer(8))
        
        self.stage_coa_3_d3 = nn.Sequential(
            nn.Conv2d(529, 529, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(529, 8, 3, padding=3, dilation=3),
            SELayer(8))
          
        self.stage_seg_2 = nn.Sequential(
            torch.nn.Conv2d(209, 209, 3, padding=1),
            torch.nn.BatchNorm2d(209),
            torch.nn.ELU(),
            torch.nn.Conv2d(209, 1, 1))

        
        self.stage_coa_2_d1 = nn.Sequential(
            nn.Conv2d(209, 209, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(209, 8, 3, padding=1, dilation=1),
            SELayer(8))

        
        self.stage_coa_2_d3 = nn.Sequential(
            nn.Conv2d(209, 209, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(209, 8, 3, padding=3, dilation=3),
            SELayer(8))
        
        self.stage_seg_1 = nn.Sequential(
            torch.nn.Conv2d(209, 209, 3, padding=1),
            torch.nn.BatchNorm2d(209),
            torch.nn.ELU(),
            torch.nn.Conv2d(209, 1, 1))
        
        self.stage_coa_1_d1 = nn.Sequential(
            nn.Conv2d(209, 209, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(209, 8, 3, padding=1, dilation=1),
            SELayer(8))
        
        self.stage_coa_1_d3 = nn.Sequential(
            nn.Conv2d(209, 209, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(209, 8, 3, padding=3, dilation=3),
            SELayer(8))

        self.se_4 = SELayer(785)
        self.se_3 = SELayer(529)
        self.se_2 = SELayer(209)
        self.se_1 = SELayer(209)
        self.se_0 = SELayer(49)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, inputs):        
        conv5 = inputs[4]
        conv4 = inputs[3]
        conv3 = inputs[2]
        conv2 = inputs[1]
        conv1 = inputs[0]
    
        pool = self.pool(conv5)
        center = self.center(pool)
        concat_5 = torch.cat([cat_non_matching(conv5, center)], dim=1)
              
        seg_5_out = self.stage_seg_5(concat_5)
        seg_5_out = self.upsample(seg_5_out)
        coa_5_d1 = self.stage_coa_5_d1(concat_5)
        coa_5_d1 = self.upsample(coa_5_d1)
        coa_5_d3 = self.stage_coa_5_d3(concat_5)
        coa_5_d3 = self.upsample(coa_5_d3)
        seg_5 = torch.cat([seg_5_out, coa_5_d1, coa_5_d3], dim=1)
        dec5 = self.dec5(concat_5)
        concat_4 = torch.cat([cat_non_matching(conv4, dec5), seg_5], dim=1)
        se_4 = self.se_4(concat_4)      
          
        seg_4_out = self.stage_seg_4(se_4)
        seg_4_out = self.upsample(seg_4_out)
        coa_4_d1 = self.stage_coa_4_d1(se_4)
        coa_4_d1 = self.upsample(coa_4_d1)
        coa_4_d3 = self.stage_coa_4_d3(se_4)
        coa_4_d3 = self.upsample(coa_4_d3)
        seg_4 = torch.cat([seg_4_out, coa_4_d1, coa_4_d3], dim=1)
        dec4 = self.dec4(se_4)     
        concat_3 = torch.cat([cat_non_matching(conv3, dec4), seg_4], dim=1)
        se_3 = self.se_3(concat_3)
        
        seg_3_out = self.stage_seg_3(se_3)
        seg_3_out = self.upsample(seg_3_out)
        coa_3_d1 = self.stage_coa_3_d1(se_3)
        coa_3_d1 = self.upsample(coa_3_d1)
        coa_3_d3 = self.stage_coa_3_d3(se_3)
        coa_3_d3 = self.upsample(coa_3_d3)
        seg_3 = torch.cat([seg_3_out, coa_3_d1, coa_3_d3], dim=1)
        dec3 = self.dec3(se_3)
        concat_2 = torch.cat([cat_non_matching(conv2, dec3), seg_3], dim=1)
        se_2 = self.se_2(concat_2)
        
        seg_2_out = self.stage_seg_2(se_2)
        seg_2_out = self.upsample(seg_2_out)
        coa_2_d1 = self.stage_coa_2_d1(se_2)
        coa_2_d1 = self.upsample(coa_2_d1)
        coa_2_d3 = self.stage_coa_2_d3(se_2)
        coa_2_d3 = self.upsample(coa_2_d3)
        seg_2 = torch.cat([seg_2_out, coa_2_d1, coa_2_d3], dim=1)
        dec2 = self.dec2(se_2)
        concat_1 = torch.cat([cat_non_matching(conv1, dec2), seg_2], dim=1)
        se_1 = self.se_1(concat_1)
        
        seg_1_out = self.stage_seg_1(se_1)
        seg_1_out = self.upsample(seg_1_out)
        coa_1_d1 = self.stage_coa_1_d1(se_1)
        coa_1_d1 = self.upsample(coa_1_d1)
        coa_1_d3 = self.stage_coa_1_d3(se_1)
        coa_1_d3 = self.upsample(coa_1_d3)
        seg_1 = torch.cat([seg_1_out, coa_1_d1, coa_1_d3], dim=1)
        dec1 = self.dec1(se_1) 
        concat_out = torch.cat([seg_1, dec1], dim=1)
        se_0 = self.se_0(concat_out)
                        
        y = F.dropout2d(se_0, p=self.dropout_2d)  
        
        result = OrderedDict()
        result['out'] = y
        result['out_512'] = seg_1_out
        result['out_256'] = seg_2_out
        result['out_128'] = seg_3_out
        result['out_64'] = seg_4_out
        result['out_32'] = seg_5_out
        
        result['out_512_d1'] = coa_1_d1  
        result['out_256_d1'] = coa_2_d1
        result['out_128_d1'] = coa_3_d1
        result['out_64_d1'] = coa_4_d1
        result['out_32_d1'] = coa_5_d1
        
        result['out_512_d3'] = coa_1_d3
        result['out_256_d3'] = coa_2_d3
        result['out_128_d3'] = coa_3_d3
        result['out_64_d3'] = coa_4_d3
        result['out_32_d3'] = coa_5_d3
        
        return result


# SwinUnetMultiCoaDecoderV2 ----------------------------------------------------------------------------------------------------------------------------------------------------------
class SwinUnetMultiCoaDecoderV2(nn.Module):
    """Optimised version of SwinUnetMultiCoaDecoder.

    Three changes vs. the original:
    1. A shared 1x1 bottleneck (C → bottleneck) replaces the expensive CxC 3x3
       convolution that was duplicated in every CoA branch.
    2. d1 and d3 CoA branches share a single bottleneck projection, halving the
       per-stage projection cost.
    3. The seg-head bottleneck also uses 1x1 instead of 3x3, keeping the same
       receptive-field trade-off but at a fraction of the params.
    """

    # Stage channels match the original decoder concatenation sizes.
    _STAGE_CHANNELS = {5: 1280, 4: 785, 3: 529, 2: 209, 1: 209}

    def __init__(self, dropout_2d: float = 0.2, is_deconv: bool = False,
                 bottleneck: int = 64):
        super().__init__()
        self.dropout_2d = dropout_2d
        self.pool = nn.MaxPool2d(2, 2)
        self.center = DecoderBlockV2(1024, 512, 256, is_deconv)
        self.dec5 = DecoderBlockV2(1280, 512, 256, is_deconv)
        self.dec4 = DecoderBlockV2(785, 512, 256, is_deconv)
        self.dec3 = DecoderBlockV2(529, 256, 64, is_deconv)
        self.dec2 = DecoderBlockV2(209, 128, 128, is_deconv)
        self.dec1 = DecoderBlockV2(209, 128, 32, is_deconv)

        B = bottleneck
        for stage, C in self._STAGE_CHANNELS.items():
            # CHANGE 2 — Seg head bottleneck:
            #   v1: Conv2d(C, C, 3×3) + BN + ELU + Conv2d(C, 1, 1×1)  ← C² params
            #   v2: Conv2d(C, B, 1×1) + BN + ELU + Conv2d(B, 1, 1×1)  ← C·B params
            #   Output is still 1 channel — compatible with CadNET's seg loss.
            setattr(self, f'stage_seg_{stage}', nn.Sequential(
                nn.Conv2d(C, B, 1), nn.BatchNorm2d(B), nn.ELU(),
                nn.Conv2d(B, 1, 1)))

            # CHANGE 1 — Shared CoA bottleneck:
            #   v1: d1 branch had Conv2d(C, C, 3×3); d3 branch had its own Conv2d(C, C, 3×3)
            #       → two separate C² projections on the same input x
            #   v2: one shared Conv2d(C, B, 1×1) feeds both d1 and d3
            #       → single C·B projection, halving the per-stage projection cost
            setattr(self, f'stage_coa_{stage}_shared', nn.Sequential(
                nn.Conv2d(C, B, 1), nn.ReLU()))

            # d1 / d3 branches now project from B channels instead of C
            # Output is still 8 channels per branch → combined = 17ch (1+8+8), same as v1
            setattr(self, f'stage_coa_{stage}_d1', nn.Sequential(
                nn.Conv2d(B, 8, 3, padding=1, dilation=1), SELayer(8)))
            setattr(self, f'stage_coa_{stage}_d3', nn.Sequential(
                nn.Conv2d(B, 8, 3, padding=3, dilation=3), SELayer(8)))

        self.se_4 = SELayer(785)
        self.se_3 = SELayer(529)
        self.se_2 = SELayer(209)
        self.se_1 = SELayer(209)
        self.se_0 = SELayer(49)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _stage(self, x, stage):
        """Run seg + CoA branches for one decoder stage and return all outputs.

        CHANGE 3 — Output shapes are identical to v1:
          seg_out : [B, 1,  2H, 2W]  (upsampled binary prediction)
          d1, d3  : [B, 8,  2H, 2W]  (upsampled CoA feature maps)
          combined: [B, 17, 2H, 2W]  (1+8+8, fed forward to next stage concat)
        The final se_0 still operates on 49ch (17+32), so CadNET's seg_module
        and all downstream loss terms remain fully compatible.
        """
        seg_out = getattr(self, f'stage_seg_{stage}')(x)
        seg_out = self.upsample(seg_out)

        # shared projection → split into d1 / d3 dilated branches
        shared = getattr(self, f'stage_coa_{stage}_shared')(x)
        d1 = self.upsample(getattr(self, f'stage_coa_{stage}_d1')(shared))
        d3 = self.upsample(getattr(self, f'stage_coa_{stage}_d3')(shared))

        combined = torch.cat([seg_out, d1, d3], dim=1)  # 1 + 8 + 8 = 17ch
        return seg_out, d1, d3, combined

    def forward(self, inputs):
        conv5 = inputs[4]
        conv4 = inputs[3]
        conv3 = inputs[2]
        conv2 = inputs[1]
        conv1 = inputs[0]

        pool = self.pool(conv5)
        center = self.center(pool)
        concat_5 = torch.cat([cat_non_matching(conv5, center)], dim=1)

        seg_5_out, coa_5_d1, coa_5_d3, seg_5 = self._stage(concat_5, 5)
        dec5 = self.dec5(concat_5)
        concat_4 = torch.cat([cat_non_matching(conv4, dec5), seg_5], dim=1)
        se_4 = self.se_4(concat_4)

        seg_4_out, coa_4_d1, coa_4_d3, seg_4 = self._stage(se_4, 4)
        dec4 = self.dec4(se_4)
        concat_3 = torch.cat([cat_non_matching(conv3, dec4), seg_4], dim=1)
        se_3 = self.se_3(concat_3)

        seg_3_out, coa_3_d1, coa_3_d3, seg_3 = self._stage(se_3, 3)
        dec3 = self.dec3(se_3)
        concat_2 = torch.cat([cat_non_matching(conv2, dec3), seg_3], dim=1)
        se_2 = self.se_2(concat_2)

        seg_2_out, coa_2_d1, coa_2_d3, seg_2 = self._stage(se_2, 2)
        dec2 = self.dec2(se_2)
        concat_1 = torch.cat([cat_non_matching(conv1, dec2), seg_2], dim=1)
        se_1 = self.se_1(concat_1)

        seg_1_out, coa_1_d1, coa_1_d3, seg_1 = self._stage(se_1, 1)
        dec1 = self.dec1(se_1)
        concat_out = torch.cat([seg_1, dec1], dim=1)
        se_0 = self.se_0(concat_out)

        y = F.dropout2d(se_0, p=self.dropout_2d)

        result = OrderedDict()
        result['out'] = y
        result['out_512'] = seg_1_out
        result['out_256'] = seg_2_out
        result['out_128'] = seg_3_out
        result['out_64'] = seg_4_out
        result['out_32'] = seg_5_out

        result['out_512_d1'] = coa_1_d1
        result['out_256_d1'] = coa_2_d1
        result['out_128_d1'] = coa_3_d1
        result['out_64_d1'] = coa_4_d1
        result['out_32_d1'] = coa_5_d1

        result['out_512_d3'] = coa_1_d3
        result['out_256_d3'] = coa_2_d3
        result['out_128_d3'] = coa_3_d3
        result['out_64_d3'] = coa_4_d3
        result['out_32_d3'] = coa_5_d3

        return result


# CoANet ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def connect_module(feature_maps, dilation_rate, out_channels):
    connect_branch = nn.Sequential(nn.Conv2d(feature_maps, feature_maps, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(feature_maps, out_channels, 3, padding=dilation_rate, dilation=dilation_rate),
                                            )
    return connect_branch

# FrameField ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
def seg_loss(pred, target):
    # Use from logits and remove all sigmoid functions: https://discuss.pytorch.org/t/bceloss-are-unsafe-to-autocast/110407/3
    # Yes, you can remove the sigmoids and work with logits by modifying the final layers and then using loss functions (like torch.nn.BCEWithLogitsLoss or a logits‐compatible version of DiceLoss) that expect raw logits. This way the model outputs won't be squashed, which is often preferred for stability during training.
    # If you switch to a “from logits” loss, you should remove the sigmoid during training. However, at inference you’ll often want to convert raw logits into probabilities (for example, to apply a threshold), so you’ll need to apply a sigmoid there if that’s what your application requires.
    if target.dim() == pred.dim() - 1:
        target = target.unsqueeze(1)
    
    loss_dice_ = smp.losses.DiceLoss(mode='binary', from_logits=True)
    loss_dice_ = loss_dice_(pred, target)
    loss_focal_ = FocalLoss(mode='binary')
    loss_focal_ = loss_focal_(pred, target)
    seg_loss = loss_dice_ + loss_focal_
    
    return seg_loss

def seg_module(backbone_features, seg_channels=1):
    seg_module = torch.nn.Sequential(
        torch.nn.Conv2d(backbone_features, backbone_features, 3, padding=1),
        torch.nn.BatchNorm2d(backbone_features),
        torch.nn.ELU(),
        torch.nn.Conv2d(backbone_features, seg_channels, 1),
        )
    
    return seg_module

def get_out_channels(module):
    if hasattr(module, 'out_channels'):
        return module.out_channels
    children = list(module.children())
    i = 1
    out_channels = None
    while out_channels is None and i <= len(children):
        last_child = children[-i]
        out_channels = get_out_channels(last_child)
        i += 1
    return out_channels

# SegFormer ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class SegFormerBackbone(nn.Module):
    """Native smp.Segformer (requires smp>=0.4.0). The SegFormer All-MLP decoder
    outputs at 1/4 input resolution; we bilinearly upsample back to full resolution
    before passing the feature maps to seg_module."""

    def __init__(self, encoder_name='mit_b2', encoder_weights='imagenet'):
        super().__init__()
        self._smp_model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.out_channels = self._smp_model.segmentation_head[0].in_channels

    def forward(self, x):
        features = self._smp_model.encoder(x)
        # smp>=0.5.0: decoder takes features as a list, not unpacked
        decoded = self._smp_model.decoder(features)
        # SegFormer decoder outputs at 1/4 resolution; upsample to match input size
        decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        return {'out': decoded}


class UnetPlusPlusBackbone(nn.Module):
    """smp.UnetPlusPlus with a configurable encoder. The decoder progressively
    upsamples 2x per stage, so the output is natively at full input resolution
    (512x512) with rich dense skip-connection feature maps."""

    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        self._smp_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.out_channels = self._smp_model.segmentation_head[0].in_channels

    def forward(self, x):
        features = self._smp_model.encoder(x)
        # smp>=0.5.0: decoder takes features as a list, not unpacked
        decoded = self._smp_model.decoder(features)
        return {'out': decoded}


# DeepLabV3+ --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class DeepLabV3PlusBackbone(nn.Module):
    """smp.DeepLabV3Plus backbone. The ASPP decoder outputs at 1/4 input resolution;
    we bilinearly upsample to full resolution before passing to seg_module."""

    def __init__(self, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        self._smp_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.out_channels = self._smp_model.segmentation_head[0].in_channels

    def forward(self, x):
        features = self._smp_model.encoder(x)
        # smp>=0.5.0: decoder takes features as a list
        decoded = self._smp_model.decoder(features)
        # ASPP decoder outputs at 1/4 resolution; upsample to match input size
        decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        return {'out': decoded}


# AerialFormer ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class AerialFormerMDCBlock(nn.Module):
    """Multi-Dilated Convolution block used in AerialFormer's MDCDecoder.

    Splits the input channel-wise into 3 groups, applies a separate dilated
    conv to each group, then fuses with a 1x1 conv + BN + activation.

    Reference: https://github.com/UARK-AICV/AerialFormer
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_cfg,
        act_cfg,
        custom_params={
            "kernel": (3, 3, 3),
            "padding": (3, 5, 7),
            "dilation": (3, 5, 7),
        },
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.custom_params = custom_params
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        SPLIT_NUM = 3

        self.layers = nn.ModuleList()

        self.pre_conv_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            bias=False,
        )
        quotient = self.in_channels // SPLIT_NUM
        reminder = self.in_channels % SPLIT_NUM
        sprit_channels = [quotient] * SPLIT_NUM
        if reminder == 1:
            sprit_channels[0] += 1
            sprit_channels[1] += 1
            sprit_channels[2] -= 1
        elif reminder == 2:
            sprit_channels[0] += 1
            sprit_channels[1] += 1
        for kernel, padding, dilation, channels in zip(
            *custom_params.values(), sprit_channels
        ):
            self.layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )

        self.fusion_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = mmcv_build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

    def forward(self, x):
        x_shape = x.shape
        x = self.pre_conv_layer(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)

        assert (
            x1.shape[1] + x2.shape[1] + x3.shape[1] == x_shape[1]
        ), f"{x1.shape[1]} + {x2.shape[1]} + {x3.shape[1]} != {x_shape[1]}"

        x1 = self.layers[0](x1)
        x2 = self.layers[1](x2)
        x3 = self.layers[2](x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion_layer(x)
        return self.act(self.norm(x))


class AerialFormerMDCDecoder(BaseDecodeHead):
    """Multi-scale MDC decoder used in AerialFormer.

    Five-stage progressive upsampling decoder that pairs each encoder level
    with a Multi-Dilated Convolution (MDC) block.  The final output is a
    sigmoid-activated segmentation map upsampled to 512×512.

    Reference: https://github.com/UARK-AICV/AerialFormer
    """

    def __init__(self):
        super().__init__(
            in_channels=[64, 128, 256, 512, 1024],
            channels=128,
            num_classes=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            input_transform="multiple_select",
            in_index=[0, 1, 2, 3, 4],
            dropout_ratio=0.1,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        )

        num_inputs = len(self.in_channels)
        self.in_channels = list(reversed(self.in_channels))
        assert num_inputs == len(self.in_index)

        self.up_convs = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()

        custom_params_list = [
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},  # deepest
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 2, 3), "dilation": (1, 2, 3)},
            {"kernel": (3, 3, 3), "padding": (1, 1, 1), "dilation": (1, 1, 1)},
            {"kernel": (1, 3, 3), "padding": (0, 1, 1), "dilation": (1, 1, 1)},  # shallowest
        ]

        for idx in range(len(self.in_channels)):
            if idx != 0:
                self.up_convs.append(
                    self._up_pooling(self.in_channels[idx - 1], self.in_channels[idx])
                )
            else:
                self.up_convs.append(nn.Identity())

            self.dilated_convs.append(
                nn.Sequential(
                    AerialFormerMDCBlock(
                        in_channels=self.in_channels[idx] * 2 ** (idx != 0),
                        out_channels=self.in_channels[idx],
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        custom_params=custom_params_list[idx],
                    ),
                    ConvModule(
                        in_channels=self.in_channels[idx],
                        out_channels=self.in_channels[idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                )
            )

        self.conv_seg = nn.Conv2d(self.in_channels[-1], self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _up_pooling(self, in_channels, out_channels, kernel_size=2, stride=2):
        """Transposed-conv upsampling block (2×) with BN + activation."""
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, bias=False,
            ),
            mmcv_build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
        )

    def forward(self, inputs):
        inputs = list(reversed(self._transform_inputs(inputs)))
        assert len(inputs) == len(self.in_index), (
            f"Expected {len(self.in_index)} inputs but got {len(inputs)}"
        )
        x = inputs[0]
        x = self.dilated_convs[0](x)
        for idx in range(1, len(inputs)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, inputs[idx]], dim=1)
            x = self.dilated_convs[idx](x)
        seg = self.cls_seg(x)
        seg = self.sigmoid(seg)
        out = F.interpolate(seg, size=(512, 512), mode='bilinear', align_corners=False)
        return out

    def forward_features(self, inputs):
        """Run the full MDC decoder path without cls_seg/sigmoid.

        Returns pre-classification feature maps [B, 64, 512, 512] in the
        {'out': features} format expected by CadNET / seg_module.
        """
        inputs = list(reversed(self._transform_inputs(inputs)))
        x = inputs[0]
        x = self.dilated_convs[0](x)
        for idx in range(1, len(inputs)):
            x = self.up_convs[idx](x)
            x = torch.cat([x, inputs[idx]], dim=1)
            x = self.dilated_convs[idx](x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        return x


class AerialFormerBackbone(nn.Module):
    """AerialFormer backbone for use inside CadNET.

    Encoder : SwinStemTransformer (Swin-Base + convolutional stem, pretrained
              on ImageNet-22k at 384×384).
    Decoder : AerialFormerMDCDecoder (5-stage Multi-Dilated CNN decoder with
              progressive transposed-conv upsampling).

    Exposes pre-classification 64-channel feature maps at full input resolution
    via {'out': features}, matching the backbone contract expected by CadNET.

    Reference: https://github.com/UARK-AICV/AerialFormer
    """

    # in_channels[-1] of MDCDecoder after channel-list reversal:
    # original [64,128,256,512,1024] reversed → last = 64
    out_channels = 64

    def __init__(self):
        super().__init__()
        self.encoder = SwinStemTransformer()
        self.encoder.init_weights()
        self.decoder = AerialFormerMDCDecoder()

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        features = self.decoder.forward_features(encoder_outputs)  # [B, 64, 512, 512]
        return {'out': features}


# CoANet -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Inlined from https://github.com/mj129/CoANet
# backbone/resnet.py
import math as _math

class CoABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class CoAResNet(nn.Module):
    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super().__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self._init_weight()
        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation, BatchNorm=BatchNorm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return e1, e2, e3, e4

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, _math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        import torch.utils.model_zoo as model_zoo
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


# aspp.py
class _CoAASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                                     padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CoAASPP(nn.Module):
    def __init__(self, output_stride=16, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        inplanes = 2048  # ResNet-101 layer4 output channels
        dilations = [1, 6, 12, 18] if output_stride == 16 else [1, 12, 24, 36]
        self.aspp1 = _CoAASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _CoAASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _CoAASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _CoAASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# decoder.py
class CoADecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super().__init__()
        self.inp = inp
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.deconv1 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        self.deconv2 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv3 = nn.Conv2d(in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0))
        self.deconv4 = nn.Conv2d(in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4))
        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self._inv_h_transform(self.deconv3(self._h_transform(x)))
        x4 = self._inv_v_transform(self.deconv4(self._v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.relu2(self.bn2(x))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _h_transform(self, x):
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def _inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., 0:shape[-2]]

    def _v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def _inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0:shape[-2]]
        return x.permute(0, 1, 3, 2)


class CoADecoder(nn.Module):
    def __init__(self, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        in_inplanes = 256  # ASPP output: 256ch fed into decoder4
        self.decoder4 = CoADecoderBlock(in_inplanes, 256, BatchNorm)
        self.decoder3 = CoADecoderBlock(512, 128, BatchNorm)
        self.decoder2 = CoADecoderBlock(256, 64, BatchNorm, inp=True)
        self.decoder1 = CoADecoderBlock(128, 64, BatchNorm, inp=True)
        self.conv_e3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False), BatchNorm(256), nn.ReLU())
        self.conv_e2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False), BatchNorm(128), nn.ReLU())
        self.conv_e1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), BatchNorm(64), nn.ReLU())
        self._init_weight()

    def forward(self, e1, e2, e3, e4):
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        d1 = self.decoder1(d2)
        return F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# connect.py
class CoASELayer(nn.Module):
    def __init__(self, channel, reduction=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CoAConnect(nn.Module):
    def __init__(self, num_classes=1, num_neighbor=9, BatchNorm=nn.BatchNorm2d, reduction=3):
        super().__init__()
        self.seg_branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1),
        )
        self.connect_branch = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_neighbor, 3, padding=1, dilation=1),
        )
        self.se = CoASELayer(num_neighbor, reduction)
        self.connect_branch_d1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_neighbor, 3, padding=3, dilation=3),
        )
        self.se_d1 = CoASELayer(num_neighbor, reduction)
        self._init_weight()

    def forward(self, x):
        seg = self.seg_branch(x)
        con0 = self.se(self.connect_branch(x))
        con1 = self.se_d1(self.connect_branch_d1(x))
        return torch.sigmoid(seg), con0, con1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CoANetBackbone(nn.Module):
    """CoANet backbone for use inside CadNET.

    Encoder : ResNet-101 (pretrained ImageNet).
    Neck     : ASPP (Atrous Spatial Pyramid Pooling).
    Decoder  : CoA decoder with orthogonal skewed-convolution blocks.

    Exposes 64-channel feature maps at full input resolution via
    {'out': features}, matching the backbone contract expected by CadNET.

    Reference: https://github.com/mj129/CoANet
    """

    out_channels = 64

    def __init__(self, output_stride=8, pretrained=True):
        super().__init__()
        BatchNorm = nn.BatchNorm2d
        self.encoder = CoAResNet(CoABottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
        self.aspp = CoAASPP(output_stride=output_stride, BatchNorm=BatchNorm)
        self.decoder = CoADecoder(BatchNorm=BatchNorm)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)
        e4 = self.aspp(e4)
        features = self.decoder(e1, e2, e3, e4)  # [B, 64, H, W]
        return {'out': features}


# CadNET ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class CadNET(BaseModel):
    def __init__(self, cfg, run_type='train', **kwargs):
        super().__init__(cfg, run_type)   
        self.cfg = cfg
        
        if self.cfg.MODEL.BACKBONE == 'unet':
            self.backbone = UNetBackbone(n_channels=3, n_hidden_base=16)
            self.backbone_out_features = get_out_channels(self.backbone)
        
        elif self.cfg.MODEL.BACKBONE == 'resunet':
            self.backbone = UNetResNetBackbone(encoder_depth=101, pretrained=True)
            self.backbone_out_features = get_out_channels(self.backbone)       
            
        elif self.cfg.MODEL.BACKBONE == 'swinunet':
            self.encoder = SwinStemTransformer()
            self.encoder.init_weights()
            self.decoder = SwinUnetDecoder()
            self.backbone_out_features = get_out_channels(self.decoder)  
            
        elif self.cfg.MODEL.BACKBONE == 'swinunet_multi_coa':
            self.encoder = SwinStemTransformer()
            self.encoder.init_weights()
            self.decoder = SwinUnetMultiCoaDecoder()
            self.backbone_out_features = 49

        elif self.cfg.MODEL.BACKBONE == 'swinunet_multi_coa_v2':
            self.encoder = SwinStemTransformer()
            self.encoder.init_weights()
            self.decoder = SwinUnetMultiCoaDecoderV2()
            self.backbone_out_features = 49

        elif self.cfg.MODEL.BACKBONE == 'segformer':
            self.backbone = SegFormerBackbone(
                encoder_name='mit_b2',
                encoder_weights='imagenet',
            )
            self.backbone_out_features = get_out_channels(self.backbone)

        elif self.cfg.MODEL.BACKBONE == 'unetplusplus':
            self.backbone = UnetPlusPlusBackbone(
                encoder_name='resnet50',
                encoder_weights='imagenet',
            )
            self.backbone_out_features = get_out_channels(self.backbone)

        elif self.cfg.MODEL.BACKBONE == 'deeplabv3plus':
            self.backbone = DeepLabV3PlusBackbone(
                encoder_name='resnet50',
                encoder_weights='imagenet',
            )
            self.backbone_out_features = get_out_channels(self.backbone)

        elif self.cfg.MODEL.BACKBONE == 'aerialformer':
            self.backbone = AerialFormerBackbone()
            self.backbone_out_features = AerialFormerBackbone.out_channels

        elif self.cfg.MODEL.BACKBONE == 'coanet':
            self.backbone = CoANetBackbone(
                output_stride=8,
                pretrained=True,
            )
            self.backbone_out_features = CoANetBackbone.out_channels

        else:
            raise ValueError(f'Invalid backbone: {self.cfg.MODEL.BACKBONE}')
        
        if self.cfg.MODEL.USE_BRK:
            if self.cfg.MODEL.USE_COA:
                self.brk_connect_d1_module_512 = connect_module(self.backbone_out_features, 1, 8)
                self.brk_connect_d3_module_512 = connect_module(self.backbone_out_features, 3, 8)
                self.brk_se1 = SELayer(8)   
                self.brk_se3 = SELayer(8)
            self.brk_seg_module_512 = seg_module(self.backbone_out_features, 1)
            
    def build_model(self, image):
        outputs = {}
        
        def create_preds(self, outputs, type_, channel_type='', channel=0, coa_start=0, coa_end=8):             
            bb_out_brk = backbone_outputs['out']
            
            #https://stackoverflow.com/questions/7129736/python-variable-method-name
            seg_func = f'{type_}_seg_module_512'            
            outputs[f'pred_bin_{type_}_512{channel_type}'] = getattr(self, seg_func)(bb_out_brk)[:,channel,:,:].unsqueeze(1)

            if self.cfg.MODEL.USE_MULTI:
                outputs[f'pred_bin_{type_}_s512{channel_type}'] = backbone_outputs['out_512'][:,channel,:,:].unsqueeze(1)
                outputs[f'pred_bin_{type_}_s256{channel_type}'] = backbone_outputs['out_256'][:,channel,:,:].unsqueeze(1)
                outputs[f'pred_bin_{type_}_s128{channel_type}'] = backbone_outputs['out_128'][:,channel,:,:].unsqueeze(1)
                outputs[f'pred_bin_{type_}_s64{channel_type}'] = backbone_outputs['out_64'][:,channel,:,:].unsqueeze(1)
                if not self.cfg.MODEL.BACKBONE.startswith('unet'):
                    outputs[f'pred_bin_{type_}_s32{channel_type}'] = backbone_outputs['out_32'][:,channel,:,:].unsqueeze(1)
            
            if self.cfg.MODEL.USE_COA: 
                d1_conn_func = f'{type_}_connect_d1_module_512'
                d1_se = f'{type_}_se1'
                d3_conn_func =f'{type_}_connect_d3_module_512'
                d3_se = f'{type_}_se3'
                outputs[f'pred_cc_d1_{type_}_512{channel_type}'] = getattr(self, d1_se)(getattr(self, d1_conn_func)(bb_out_brk))[:,coa_start:coa_end,:,:]
                outputs[f'pred_cc_d3_{type_}_512{channel_type}'] = getattr(self, d3_se)(getattr(self, d3_conn_func)(bb_out_brk))[:,coa_start:coa_end,:,:]
                
                if self.cfg.MODEL.USE_MULTI:
                    outputs[f'pred_cc_d1_{type_}_s512{channel_type}'] = backbone_outputs['out_512_d1'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d3_{type_}_s512{channel_type}'] = backbone_outputs['out_512_d3'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d1_{type_}_s256{channel_type}'] = backbone_outputs['out_256_d1'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d3_{type_}_s256{channel_type}'] = backbone_outputs['out_256_d3'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d1_{type_}_s128{channel_type}'] = backbone_outputs['out_128_d1'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d3_{type_}_s128{channel_type}'] = backbone_outputs['out_128_d3'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d1_{type_}_s64{channel_type}'] = backbone_outputs['out_64_d1'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d3_{type_}_s64{channel_type}'] = backbone_outputs['out_64_d3'][:,coa_start:coa_end,:,:]
                    
                    if not self.cfg.MODEL.BACKBONE.startswith('unet'):
                        outputs[f'pred_cc_d1_{type_}_s32{channel_type}'] = backbone_outputs['out_32_d1'][:,coa_start:coa_end,:,:]
                        outputs[f'pred_cc_d3_{type_}_s32{channel_type}'] = backbone_outputs['out_32_d3'][:,coa_start:coa_end,:,:]
        
        _smp_backbones = ('segformer', 'unetplusplus', 'deeplabv3plus', 'aerialformer', 'coanet')
        if self.cfg.MODEL.BACKBONE.startswith('unet') or 'resunet' in self.cfg.MODEL.BACKBONE or self.cfg.MODEL.BACKBONE in _smp_backbones:
            backbone_outputs = self.backbone(image)
        else:
            encoder_outputs = self.encoder(image)
            backbone_outputs = self.decoder(encoder_outputs)
                
        if self.cfg.MODEL.USE_BRK:   
            create_preds(self, outputs, 'brk', channel_type='', channel=0, coa_start=0, coa_end=8)

        return outputs
    
    def forward(self, images, annotations, norm=False):
        if self.run_type == 'train':
            return self.forward_train(images, annotations, norm)
        elif self.run_type == 'predict':
            return self.forward_predict(images, annotations)

    def forward_train(self, images, annotations, norm=False):        
        pred = self.build_model(images)  

        loss_dict = {
                'seg_edge_loss_brk_512': None,
                'seg_edge_loss_brk_s512': None,
                'seg_edge_loss_brk_s256': None,
                'seg_edge_loss_brk_s128': None,
                'seg_edge_loss_brk_s64': None,
                'seg_edge_loss_brk_s32': None,
                
                'seg_edge_coa_loss_brk_512': None,
                'seg_edge_coa_loss_brk_s512': None,
                'seg_edge_coa_loss_brk_s256': None,
                'seg_edge_coa_loss_brk_s128': None,
                'seg_edge_coa_loss_brk_s64': None,
                'seg_edge_coa_loss_brk_s32': None,
                }
                
        if self.cfg.MODEL.USE_BRK:
            gt_brk_bin_512 = annotations['gt_bin_brk_512']
            pred_brk_bin_512 = pred['pred_bin_brk_512']
            edge_loss_brk_512 = seg_loss(pred_brk_bin_512, gt_brk_bin_512)
            loss_dict['seg_edge_loss_brk_512'] = edge_loss_brk_512

            if self.cfg.MODEL.USE_MULTI:
                edge_loss_brk_s512 = seg_loss(pred['pred_bin_brk_s512'], gt_brk_bin_512)
                loss_dict['seg_edge_loss_brk_s512'] = edge_loss_brk_s512

                edge_loss_brk_s256 = seg_loss(pred['pred_bin_brk_s256'], annotations['gt_bin_brk_256_visibility'][:,0,:,:].unsqueeze(1))
                loss_dict['seg_edge_loss_brk_s256'] = edge_loss_brk_s256
                
                edge_loss_brk_s128 = seg_loss(pred['pred_bin_brk_s128'], annotations['gt_bin_brk_128_visibility'][:,0,:,:].unsqueeze(1))
                loss_dict['seg_edge_loss_brk_s128'] = edge_loss_brk_s128

                
                edge_loss_brk_s64 = seg_loss(pred['pred_bin_brk_s64'], annotations['gt_bin_brk_64_visibility'][:,0,:,:].unsqueeze(1))
                loss_dict['seg_edge_loss_brk_s64'] = edge_loss_brk_s64
                
                if not self.cfg.MODEL.BACKBONE.startswith('unet'):
                    edge_loss_brk_512_s32 = seg_loss(pred['pred_bin_brk_s32'], annotations['gt_bin_brk_32_visibility'][:,0,:,:].unsqueeze(1))
                    loss_dict['seg_edge_loss_brk_s32'] = edge_loss_brk_512_s32
            
            if self.cfg.MODEL.USE_COA:
                pred_connect_d1_brk_512 = pred['pred_cc_d1_brk_512']
                pred_connect_d3_brk_512 = pred['pred_cc_d3_brk_512']

                loss_connect_d1_brk_512 = seg_loss(pred_connect_d1_brk_512, annotations[f'gt_cc_d1_brk_512'])
                loss_connect_d3_brk_512 = seg_loss(pred_connect_d3_brk_512, annotations[f'gt_cc_d3_brk_512'])  
                coa_loss_brk_512 = self.cfg.WEIGHTS.COA_D1_LOSS * loss_connect_d1_brk_512 + self.cfg.WEIGHTS.COA_D3_LOSS * loss_connect_d3_brk_512
                loss_dict['seg_edge_coa_loss_brk_512'] = coa_loss_brk_512

                if self.cfg.MODEL.USE_MULTI:
                    loss_connect_d1_brk_s512 = seg_loss(pred['pred_cc_d1_brk_s512'], annotations[f'gt_cc_d1_brk_512'])
                    loss_connect_d3_brk_s512 = seg_loss(pred['pred_cc_d3_brk_s512'], annotations[f'gt_cc_d3_brk_512'])
                    coa_loss_brk_s512 = self.cfg.WEIGHTS.COA_D1_LOSS * loss_connect_d1_brk_s512 + self.cfg.WEIGHTS.COA_D3_LOSS * loss_connect_d3_brk_s512
                    loss_dict['seg_edge_coa_loss_brk_s512'] = coa_loss_brk_s512
                    
                    loss_connect_d1_brk_s256 = seg_loss(pred['pred_cc_d1_brk_s256'], annotations[f'gt_cc_d1_brk_256'])
                    loss_connect_d3_brk_s256 = seg_loss(pred['pred_cc_d3_brk_s256'], annotations[f'gt_cc_d3_brk_256'])
                    coa_loss_brk_s256 = self.cfg.WEIGHTS.COA_D1_LOSS * loss_connect_d1_brk_s256 + self.cfg.WEIGHTS.COA_D3_LOSS * loss_connect_d3_brk_s256
                    loss_dict['seg_edge_coa_loss_brk_s256'] = coa_loss_brk_s256
                    
                    loss_connect_d1_brk_s128 = seg_loss(pred['pred_cc_d1_brk_s128'], annotations[f'gt_cc_d1_brk_128'])
                    loss_connect_d3_brk_s128 = seg_loss(pred['pred_cc_d3_brk_s128'], annotations[f'gt_cc_d3_brk_128'])
                    coa_loss_brk_s128 = self.cfg.WEIGHTS.COA_D1_LOSS * loss_connect_d1_brk_s128 + self.cfg.WEIGHTS.COA_D3_LOSS * loss_connect_d3_brk_s128
                    loss_dict['seg_edge_coa_loss_brk_s128'] = coa_loss_brk_s128
                    
                    loss_connect_d1_brk_s64 = seg_loss(pred['pred_cc_d1_brk_s64'], annotations[f'gt_cc_d1_brk_64'])
                    loss_connect_d3_brk_s64 = seg_loss(pred['pred_cc_d3_brk_s64'], annotations[f'gt_cc_d3_brk_64'])
                    coa_loss_brk_s64 = self.cfg.WEIGHTS.COA_D1_LOSS * loss_connect_d1_brk_s64 + self.cfg.WEIGHTS.COA_D3_LOSS * loss_connect_d3_brk_s64
                    loss_dict['seg_edge_coa_loss_brk_s64'] = coa_loss_brk_s64

                    if not self.cfg.MODEL.BACKBONE.startswith('unet'):
                        loss_connect_d1_brk_s32 = seg_loss(pred['pred_cc_d1_brk_s32'], annotations[f'gt_cc_d1_brk_32'])
                        loss_connect_d3_brk_s32 = seg_loss(pred['pred_cc_d3_brk_s32'], annotations[f'gt_cc_d3_brk_32'])
                        coa_loss_brk_s32 = self.cfg.WEIGHTS.COA_D1_LOSS * loss_connect_d1_brk_s32 + self.cfg.WEIGHTS.COA_D3_LOSS * loss_connect_d3_brk_s32
                        loss_dict['seg_edge_coa_loss_brk_s32'] = coa_loss_brk_s32
                                  
        if self.cfg.MODEL.USE_BRK:
            gt_bin = gt_brk_bin_512
            if self.cfg.MODEL.USE_COA:
                pred_final = torch.cat((pred_brk_bin_512, pred_connect_d1_brk_512, pred_connect_d3_brk_512), dim=1).squeeze()
                pred_final = torch.amax(pred_final, axis=1)
            else: pred_final = pred_brk_bin_512
            
        recall, precision, f1 = self.calculate_metrics(gt_bin, pred_final)

        return loss_dict, recall, precision, f1       

    def forward_predict(self, images, annotations):
        preds = self.build_model(images)
        for key, value in preds.items():
            preds[key] = torch.sigmoid(value)
        return preds
