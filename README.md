# CTformer
[![Try In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bfNNZigUvlgIJ2QnD70kgLmcz1CZcInc?usp=sharing)

This repository includes implementation of TED-Net: Convolution-free T2T Vision Transformer-based Encoder-decoder Dilation network for Low-dose CT Denoising in https://arxiv.org/abs/2106.04650. This respository is originated from https://github.com/SSinyu/RED-CNN and https://github.com/yitu-opensource/T2T-ViT.

<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/153113397-bc7b93a9-a694-4b92-8ebc-fce897ddf458.png" width="700">
</p>
<!-- ![image](https://user-images.githubusercontent.com/23077770/153112136-c0ea4564-3ac8-4786-adbb-6a4252a6e37e.png) -->
<!-- ![image](https://user-images.githubusercontent.com/23077770/153113397-bc7b93a9-a694-4b92-8ebc-fce897ddf458.png) -->



**Data Preparation:**
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic https://www.aapm.org/GrandChallenge/LowDoseCT/, please refer to https://github.com/SSinyu/RED-CNN for more detailed data preparation. 

The path of .npy files for training and testing is T2T_main.py --save_path ['../aapm_all_npy_3mm/']

**Model Training and Testing:**
1. run python T2T_main.py to train. 
2. run python T2T_main.py --mode test --test_iters [set iters] to test.

**Simple Demo**
```
from t2t_shortcuts import TED_Net
import torch

x = torch.randn(1,1,64,64)
TEDNet = TED_Net(img_size=64,tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)
y = TEDNet(x)
print(y.shape)
```

**Experiment Results:**
<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/153113718-bac6dada-0a06-4006-8aa5-2d315f87ad0e.png" width="400">
</p>
<!-- ![image](https://user-images.githubusercontent.com/23077770/153113718-bac6dada-0a06-4006-8aa5-2d315f87ad0e.png) -->


<!-- <img src="https://user-images.githubusercontent.com/23077770/130271899-1e01f3c8-a4bc-46da-a9ae-4db159905eff.png" width="600">
<img src="https://user-images.githubusercontent.com/23077770/130271852-dcd9703f-9734-43f0-825c-6bb964d1f133.png" width="600"> -->

