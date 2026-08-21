import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import warnings

from collections import OrderedDict
from einops import rearrange
from models.basemodel import BaseModel, FocalLoss
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer

from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_, trunc_normal_init)
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList, load_state_dict)
from mmcv.utils import to_2tuple
from mmseg.models.backbones.swin import SwinBlockSequence
from mmseg.utils import get_root_logger
from mmseg.models.utils.embed import PatchEmbed, PatchMerging

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
            setattr(self, f'stage_seg_{stage}', nn.Sequential(
                nn.Conv2d(C, B, 1), nn.BatchNorm2d(B), nn.ELU(),
                nn.Conv2d(B, 1, 1)))
            setattr(self, f'stage_coa_{stage}_shared', nn.Sequential(
                nn.Conv2d(C, B, 1), nn.ReLU()))
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
        seg_out = getattr(self, f'stage_seg_{stage}')(x)
        seg_out = self.upsample(seg_out)

        shared = getattr(self, f'stage_coa_{stage}_shared')(x)
        d1 = self.upsample(getattr(self, f'stage_coa_{stage}_d1')(shared))
        d3 = self.upsample(getattr(self, f'stage_coa_{stage}_d3')(shared))

        combined = torch.cat([seg_out, d1, d3], dim=1)
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

# CadNET ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class CadNET(BaseModel):
    def __init__(self, cfg, run_type='train', **kwargs):
        super().__init__(cfg, run_type)   
        self.cfg = cfg
        
        self.encoder = SwinStemTransformer()
        self.encoder.init_weights()
        self.decoder = SwinUnetMultiCoaDecoderV2()
        self.backbone_out_features = 49
        
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
                    
                    outputs[f'pred_cc_d1_{type_}_s32{channel_type}'] = backbone_outputs['out_32_d1'][:,coa_start:coa_end,:,:]
                    outputs[f'pred_cc_d3_{type_}_s32{channel_type}'] = backbone_outputs['out_32_d3'][:,coa_start:coa_end,:,:]
        
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
