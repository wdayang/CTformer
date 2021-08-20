# TED-net
This repository includes implementation of TED-net: Convolution-free T2T Vision Transformer-based Encoder-decoder Dilation network for Low-dose CT Denoising in https://arxiv.org/abs/2106.04650. This respository is originated from https://github.com/SSinyu/RED-CNN and https://github.com/yitu-opensource/T2T-ViT.

![image](https://user-images.githubusercontent.com/23077770/130271069-01590029-dc3c-4b0d-96c3-3e2cdf71dc27.png)

Data preparation:
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic
(I can't share this data, you should ask at the URL below if you want)
https://www.aapm.org/GrandChallenge/LowDoseCT/

1. run python main.py to training. .
2. run python main.py --mode test --test_iters [set iters] to test.
