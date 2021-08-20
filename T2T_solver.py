import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
#from networks import RED_CNN
#from t2t_vit_v2 import T2T_ViT
from t2t_shortcuts import T2T_ViT
#from t2t_vit_full_token_1depth_1024_dilation_shift_noshort import T2T_ViT
#from t2t_ablation_nodilation import T2T_ViT

from measure import compute_measure

# =============================================================================
# from thop import profile ## params and MACs
# from thop import clever_format
# from ptflops import get_model_complexity_info
# =============================================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_arr(arr,patch_size,stride=32):    ## 512*512 to 32*32
    pad = (16, 16, 16, 16) # pad by (0, 1), (2, 1), and (3, 3)
    arr = nn.functional.pad(arr, pad, "constant", 0)
    _,_,h,w = arr.shape
    num = h//stride - 1
    arrs = torch.zeros(num*num,1,patch_size,patch_size)

    for i in range(num):
        for j in range(num):
            arrs[i*num+j,0] = arr[0,0,i*stride:i*stride+patch_size,j*stride:j*stride+patch_size]
    return arrs

def agg_arr(arrs, size, stride=32):  ## from 32*32 to size 512*512
    arr = torch.zeros(size, size)
    n,_,h,w = arrs.shape
    num = size//stride
    for i in range(num):
        for j in range(num):
            arr[i*stride:(i+1)*stride,j*stride:(j+1)*stride] = arrs[i*num+j,:,16:48,16:48]
  #return arr
    return arr.unsqueeze(0).unsqueeze(1)

class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size

        #self.REDCNN = RED_CNN()
        #self.REDCNN = T2T_ViT(tokens_type='convolution', embed_dim=768, depth=20, num_heads=12, mlp_ratio=2.).to(self.device)
        self.REDCNN = T2T_ViT(img_size=64,tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=4, stride=4, mlp_ratio=2., token_dim=64)
        #self.REDCNN = T2T_ViT(img_size=128,tokens_type='convolution', in_chans=8,embed_dim=768, depth=6, num_heads=12, kernel=16, stride=8, mlp_ratio=2.)
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)   ## data parallel  ,device_ids=[2,3]
        self.REDCNN.to(self.device)

        self.lr = args.lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.REDCNN.parameters(), self.lr)


    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'T2T_vit_{}iter.ckpt'.format(iter_))
        torch.save(self.REDCNN.state_dict(), f)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'T2T_vit_{}iter.ckpt'.format(iter_))
# =============================================================================
#         if self.multi_gpu:    ## has problem use multiple GPU testing
#             state_d = OrderedDict()
#             for k, v in torch.load(f):
#                 n = k[7:]
#                 state_d[n] = v
#             self.REDCNN.load_state_dict(state_d)
#         else:
# =============================================================================
        self.REDCNN.load_state_dict(torch.load(f))


    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        NumOfParam = count_parameters(self.REDCNN)
        print('trainable parameter:', NumOfParam)
        
# =============================================================================
#         inputx = torch.randn(1, 1, 64, 64).float().to(self.device)
#         macs, params = profile(self.REDCNN, inputs=(inputx,))
#         macs, params = clever_format([macs, params], "%.3f")
#         print("MACs:", macs)
#         print("params:", params)
# =============================================================================
# =============================================================================
#         macs, params = get_model_complexity_info(self.REDCNN, (1, 64, 64), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#         print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#         print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# =============================================================================

        train_losses = []
        total_iters = 0
        start_time = time.time()
        loss_all = []
        for epoch in range(1, self.num_epochs):
            self.REDCNN.train(True)

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)   ## expand one dimension given the dimension 0  4->[1,4]
                y = y.unsqueeze(0).float().to(self.device)   ## copy data to device

                if self.patch_size: # patch training
                    x = x.view(-1, 1, self.patch_size, self.patch_size)  ## similar to reshape
                    y = y.view(-1, 1, self.patch_size, self.patch_size)

                pred = self.REDCNN(x)
                #print(pred.shape)
                loss = self.criterion(pred, y)*100 + 1e-4  ## to prevent 0
                self.REDCNN.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                loss_all.append(loss.item())
                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(total_iters, epoch, 
                                                                                                        self.num_epochs, iter_+1, 
                                                                                                        len(self.data_loader), loss.item(), 
                                                                                                    time.time() - start_time))
                # learning rate decay
                #print(total_iters)  
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                if total_iters % 20000 == 0:
                    print("save model: ",total_iters)
                    self.save_model(total_iters)
                    np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
        self.save_model(total_iters)
        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
        print("total_iters:",total_iters)
        ## save loss figure
        plt.plot(np.array(loss_all), 'r')  ## print out the loss curve
        #plt.show()
        plt.savefig('save/loss.png')

    def test(self):
        del self.REDCNN
        # load
        #self.REDCNN = RED_CNN().to(self.device)
        #self.REDCNN = T2T_ViT(tokens_type='convolution', embed_dim=768, depth=20, num_heads=12, mlp_ratio=2.)
        self.REDCNN = T2T_ViT(img_size=64,tokens_type='performer', embed_dim=512, depth=1, num_heads=8, kernel=8, stride=4, mlp_ratio=2., token_dim=64)
        #self.REDCNN = T2T_ViT(img_size=128,tokens_type='convolution', in_chans=8,embed_dim=768, depth=6, num_heads=12, kernel=16, stride=8, mlp_ratio=2.)
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            #print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.REDCNN = nn.DataParallel(self.REDCNN)   ## data parallel
        self.REDCNN.to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                #y = x-y#pred = self.REDCNN(x)
# =============================================================================
#                 arrs = split_arr(x, 64)  ## split   for image patch test
#                 arrs_pred = torch.zeros(arrs.shape)
#                 for i in range(4):   ## split into 8 parts to ease the cuda memory
#                     temp_arr = arrs[i*64:(i+1)*64].to(self.device)
#                     pred_tmp = self.REDCNN(temp_arr)
#                     arrs_pred[i*64:(i+1)*64] = pred_tmp
#                 #pred = self.REDCNN(arrs)
#                 pred = agg_arr(arrs_pred, 512)
# =============================================================================
                #pred = self.REDCNN(x)
# =============================================================================
#                 arrs = split_arr(x, 64).to(self.device)  ## split   for image patch test
#                 arrs_pred = torch.zeros(arrs.shape).to(self.device)
#                 for i in range(4):   ## split into 4 parts to ease the cuda memory   ## for operation would not work, cannot save figure
#                     temp_arr = arrs[i*64:(i+1)*64].to(self.device)
#                     pred_tmp = self.REDCNN(temp_arr)
#                     arrs_pred[i*64:(i+1)*64] = pred_tmp
#                 #pred = self.REDCNN(arrs)
#                 pred = agg_arr(arrs_pred, 512)
# =============================================================================
                
                arrs = split_arr(x, 64).to(self.device)  ## split to image patches for test into 4 patches
                arrs[0:64] = self.REDCNN(arrs[0:64])
                arrs[64:2*64] = self.REDCNN(arrs[64:2*64])
                arrs[2*64:3*64] = self.REDCNN(arrs[2*64:3*64])
                arrs[3*64:4*64] = self.REDCNN(arrs[3*64:4*64])
                pred = agg_arr(arrs, 512).to(self.device)
                
# =============================================================================
#                 arrs = split_arr(x, 64).to(self.device)  ## split
#                 pred = self.REDCNN(arrs)
#                 pred = agg_arr(pred, 512)
# =============================================================================

                #pred = x - pred# denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)

                printProgressBar(i, len(self.data_loader),
                                 prefix="Compute measurements ..",
                                 suffix='Complete', length=25)
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader), 
                                                                                            ori_ssim_avg/len(self.data_loader), 
                                                                                            ori_rmse_avg/len(self.data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader), 
                                                                                                  pred_ssim_avg/len(self.data_loader), 
                                                                                                  pred_rmse_avg/len(self.data_loader)))