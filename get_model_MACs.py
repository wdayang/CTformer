#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 16:13:39 2021

@author: dayang
"""

from t2t_ablation_shortcuts import T2T_ViT
import torch
from ptflops import get_model_complexity_info
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = T2T_ViT(img_size=64,tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)

net = net.to(device)
macs, params = get_model_complexity_info(net, (1, 64, 64), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))