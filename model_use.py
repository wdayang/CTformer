#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:13:57 2021

@author: dayang
"""
from t2t_shortcuts import TED_Net
import torch

x = torch.randn(1,1,64,64)
TEDNet = TED_Net(img_size=64,tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)
y = TEDNet(x)
print(y.shape)