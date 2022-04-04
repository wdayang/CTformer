# CTformer
[![Try In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bfNNZigUvlgIJ2QnD70kgLmcz1CZcInc?usp=sharing)

[ArXiv(extension)](https://arxiv.org/abs/2202.13517) | [ArXiv(conference)](https://arxiv.org/abs/2106.04650)

This repository includes implementation of CTformer: Convolution-free Token2Token Dilated Vision Transformer for Low-dose CT Denoising in https://arxiv.org/abs/2202.13517 and TED-Net: https://arxiv.org/abs/2106.04650. This respository is originated from https://github.com/SSinyu/RED-CNN and https://github.com/yitu-opensource/T2T-ViT.

<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/156230081-cf5488f3-14e9-4eae-bdb1-e00d6fce7527.png" width="420">
</p>
<p align="center">
  <em>Fig. 1: The architecture of the CTformer.</em>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/156230454-cf17ce76-9a93-4ffe-b165-c0b35705ef28.png" width="800">
</p>
<p align="center">
  <em>Fig. 2: The micro structures of the CTformer.</em>
</p>

<!-- ![image](https://user-images.githubusercontent.com/23077770/156230454-cf17ce76-9a93-4ffe-b165-c0b35705ef28.png) -->
<!-- ![image](https://user-images.githubusercontent.com/23077770/156230081-cf5488f3-14e9-4eae-bdb1-e00d6fce7527.png) -->
<!-- ![image](https://user-images.githubusercontent.com/23077770/153112136-c0ea4564-3ac8-4786-adbb-6a4252a6e37e.png) -->
<!-- ![image](https://user-images.githubusercontent.com/23077770/153113397-bc7b93a9-a694-4b92-8ebc-fce897ddf458.png) -->



## Data Preparation:
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic https://www.aapm.org/GrandChallenge/LowDoseCT/, please refer to https://github.com/SSinyu/RED-CNN for more detailed data preparation. 

The path of .npy files for training and testing can set in 'main.py --save_path ['../aapm_all_npy_3mm/']'

## Model Training and Testing:
```
>> python main.py  ## train CTformer. 
>> python main.py --mode test --test_iters [set iters]  ## run test.
```
## Usage Demo
```
from CTformer import CTformer
import torch

x = torch.randn(1,1,64,64)
CT_former = CTformer(img_size=64,tokens_type='performer', embed_dim=64, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)
y = CT_former(x)
print(y.shape)
```

## Experiment Results:
<p align="center">
Tab. 1: Quantitative results.
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/156231489-6a73924a-b37e-49f2-8451-570f765e7692.png" width="450">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/156231907-4f4dbcf3-1ac8-48b9-87c7-40438ef74a79.png" width="450">
</p>
<p align="center">
Fig. 3: (a) LDCT, (b) RED-CNN, (c) WGAN-VGG, (d) MAP-NN, (e) AD-NET, (f) the proposed CTformer, and (g) NDCT.
</p>
<!-- ![image](https://user-images.githubusercontent.com/23077770/156231907-4f4dbcf3-1ac8-48b9-87c7-40438ef74a79.png) -->


<!-- ![image](https://user-images.githubusercontent.com/23077770/156231489-6a73924a-b37e-49f2-8451-570f765e7692.png) -->
<!-- ![image](https://user-images.githubusercontent.com/23077770/153113718-bac6dada-0a06-4006-8aa5-2d315f87ad0e.png) -->
<!-- <img src="https://user-images.githubusercontent.com/23077770/130271899-1e01f3c8-a4bc-46da-a9ae-4db159905eff.png" width="600"> -->
<!-- <img src="https://user-images.githubusercontent.com/23077770/130271852-dcd9703f-9734-43f0-825c-6bb964d1f133.png" width="600"> -->

## Visual Interpretation:

<!-- ![image](https://user-images.githubusercontent.com/23077770/156232817-0b28b49c-1c0f-480c-b0a5-e03988e8806d.png) -->
<p align="center">
  <img src="https://user-images.githubusercontent.com/23077770/156232817-0b28b49c-1c0f-480c-b0a5-e03988e8806d.png" width="800">
</p>
<p align="center">
  <em>Fig. 4: The attention maps of the CTformer.</em>
</p>
