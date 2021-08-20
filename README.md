# TED-net
This repository includes implementation of TED-net: Convolution-free T2T Vision Transformer-based Encoder-decoder Dilation network for Low-dose CT Denoising in https://arxiv.org/abs/2106.04650. This respository is originated from https://github.com/SSinyu/RED-CNN and https://github.com/yitu-opensource/T2T-ViT.

![image](https://user-images.githubusercontent.com/23077770/130271382-15a2c5d7-b456-4537-95f2-f2870484fbfd.png)

Data preparation:
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic, please refer to https://github.com/SSinyu/RED-CNN for more detailed data preparation. 
https://www.aapm.org/GrandChallenge/LowDoseCT/

1. run python main.py to training. .
2. run python main.py --mode test --test_iters [set iters] to test.
