# TED-net
This repository includes implementation of TED-net: Convolution-free T2T Vision Transformer-based Encoder-decoder Dilation network for Low-dose CT Denoising in https://arxiv.org/abs/2106.04650. This respository is originated from https://github.com/SSinyu/RED-CNN and https://github.com/yitu-opensource/T2T-ViT.

<img src="https://user-images.githubusercontent.com/23077770/130271382-15a2c5d7-b456-4537-95f2-f2870484fbfd.png" width="600">

**Data Preparation:**
The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic https://www.aapm.org/GrandChallenge/LowDoseCT/, please refer to https://github.com/SSinyu/RED-CNN for more detailed data preparation. 

**Model Training and Testing:**
1. run python T2T_main.py to train. 
2. run python T2T_main.py --mode test --test_iters [set iters] to test.

**Experiment Results:**

<img src="https://user-images.githubusercontent.com/23077770/130271899-1e01f3c8-a4bc-46da-a9ae-4db159905eff.png" width="600">

<img src="https://user-images.githubusercontent.com/23077770/130271852-dcd9703f-9734-43f0-825c-6bb964d1f133.png" width="600">

