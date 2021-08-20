"""
T2T-ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from token_transformer import Token_transformer    ## currently did not use this part
from token_performer import Token_performer
from T2T_transformer_block import Block, get_sinusoid_encoding


class MultiHeadDense(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch, out_ch))
    
    def forward(self, x):
        # x:[b, h*w, d]
        # x = torch.bmm(x, self.weight)
        x = F.linear(x, self.weight)
        return x

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=64, tokens_type='performer', in_chans=1, embed_dim=256, token_dim=64, kernel=32, stride=32):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1),dilation=(2,2))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
        #if tokens_type == 'convolution':  # just for comparison with conolution, not our model
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(1, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        #self.num_patches = (img_size // (1 * 2 * 2)) * (img_size // (1 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately
        self.num_patches = 529   ## calculate myself
        
    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x)   ## [1, 128, 64, 128])
        #res0 = x
        #print('soft_split0:',x.shape)
        # iteration1: re-structurization/reconstruction
        x = self.attention1(x.transpose(1, 2))
        B, new_HW, C = x.shape
        #print('attention1',x.shape)
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        #print('transpose(1,2):',x.shape)
        # iteration1: soft split
        x = torch.roll(x, shifts=(-2, -2), dims=(2, 3))  ##  shift some position
        x = self.soft_split1(x)
        res_11 = x
        #print('soft_split1(x).transpose(1, 2):',x.shape)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x.transpose(1, 2))
        
        #print('attention2:',x.shape)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        
        x = torch.roll(x, shifts=(2, 2), dims=(2, 3))  ## shift back position
        #print('x.transpose(1, 2):',x.shape)
        # iteration2: soft split
        x = self.soft_split2(x)
        res_22 = x
        #res2 = x
        #print('soft_split2:',x.shape)
        # final tokens
        x = self.project(x.transpose(1, 2))  ## no projection
        #print('project',x.shape)

        return x,res_11,res_22 #,res0,res2
    
class Token_back_Image(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=64, tokens_type='performer', in_chans=1, embed_dim=256, token_dim=64, kernel=32, stride=32):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Fold((64,64),kernel_size=(7, 7), stride=(2, 2))
            self.soft_split1 = nn.Fold((29,29),kernel_size=(3, 3), stride=(1, 1),dilation=(2,2))
            self.soft_split2 = nn.Fold((25,25),kernel_size=(3, 3), stride=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.project = nn.Linear(embed_dim,token_dim * 3 * 3)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
        #if tokens_type == 'convolution':  # just for comparison with conolution, not our model
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(1, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (1 * 2 * 2)) * (img_size // (1 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x, res_11,res_22):
        # step0: soft split
        
        x = self.project(x).transpose(1, 2) #+ res2
        x = x + res_22
        x = self.soft_split2(x)
        x = torch.roll(x, shifts=(-2, -2), dims=(-1, -2))
        x = rearrange(x,'b c h w -> b c (h w)').transpose(1,2)
        x = self.attention2(x).transpose(1, 2)
        x = x + res_11
        x = self.soft_split1(x)
        x = torch.roll(x, shifts=(2, 2), dims=(-1, -2))
        
        x = rearrange(x,'b c h w -> b c (h w)').transpose(1,2)
        x = self.attention1(x).transpose(1, 2) #+ res0
        x = self.soft_split0(x) 

        return x

class T2T_ViT(nn.Module):
    def __init__(self, img_size=512, tokens_type='convolution', in_chans=1, num_classes=1000, embed_dim=768, depth=12,  ## transformer depth 12
                 num_heads=12, kernel=32, stride=32, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, token_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(    ## use module 2
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim,kernel=kernel, stride=stride)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.dconv1 = Token_back_Image(img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim, kernel=kernel, stride=stride)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        res1 = x
        #x = self.forward_features(x)
        
        B = x.shape[0]
        x, res_11, res_22 = self.tokens_to_token(x)
        res_0 = x

        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        i = 0
        for blk in self.blocks:
            i += 1
            x = blk(x)
            
            if i == 1:
                res_1 = x
            elif i == 2:
                res_2 = x
            elif i == 3:
                res_3 = x
            elif i == 4:
                res_4 = x
            elif i == 5:
                res_5 = x
             

        x = self.norm(x) #+ res_0   ## do not use 0,2,4
        #return x#,res0,res2
        
        out = res1 - self.dconv1(x,res_11,res_22)
        #print(out.shape)
        return out

