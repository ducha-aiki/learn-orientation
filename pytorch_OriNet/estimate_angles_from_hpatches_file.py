from __future__ import division, print_function
import os
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
import time
USE_CUDA = False
PS = 28
class GHH(nn.Module):
    def __init__(self, n_in, n_out, s = 4, m = 4):
        super(GHH, self).__init__()
        self.n_out = n_out
        self.s = s
        self.m = m
        self.conv = nn.Linear(n_in, n_out * s * m)
        d = torch.arange(0, s)
        self.deltas = -1.0 * (d % 2 != 0).float()  + 1.0 * (d % 2 == 0).float()
        self.deltas = Variable(self.deltas)
        return
    def forward(self,x):
        x_feats = self.conv(x.view(x.size(0),-1)).view(x.size(0), self.n_out, self.s, self.m);
        max_feats = x_feats.max(dim = 3)[0];
        if x.is_cuda:
            self.deltas = self.deltas.cuda()
        else:
            self.deltas = self.deltas.cpu()
        out =  (max_feats * self.deltas.view(1,1,-1).expand_as(max_feats)).sum(dim = 2)
        return out

class YiNet(nn.Module):
    def __init__(self, PS = 28):
        super(YiNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=0, bias = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=0, bias = True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=2, padding = 2),
            nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=0, bias = True),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            GHH(50, 100),
            GHH(100, 2)
        )
        self.input_mean = 0.427117081207483
        self.input_std = 0.21888339179665006;
        self.PS = PS
        return
    def import_weights(self, dir_name):
        self.features[0].weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer0_W.npy'))).float()
        self.features[0].bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer0_b.npy'))).float().view(-1)
        self.features[3].weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer1_W.npy'))).float()
        self.features[3].bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer1_b.npy'))).float().view(-1)
        self.features[6].weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer2_W.npy'))).float()
        self.features[6].bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer2_b.npy'))).float().view(-1)
        self.features[9].conv.weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer3_W.npy'))).float().view(50, 1600).contiguous().t().contiguous()#.view(1600, 50, 1, 1).contiguous()
        self.features[9].conv.bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer3_b.npy'))).float().view(1600)
        self.features[10].conv.weight.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer4_W.npy'))).float().view(100, 32).contiguous().t().contiguous()#.view(32, 100, 1, 1).contiguous()
        self.features[10].conv.bias.data = torch.from_numpy(np.load(os.path.join(dir_name, 'layer4_b.npy'))).float().view(32)
        self.input_mean = float(np.load(os.path.join(dir_name, 'input_mean.npy')))
        self.input_std = float(np.load(os.path.join(dir_name, 'input_std.npy')))
        return
    def input_norm1(self,x):
        return (x - self.input_mean) / self.input_std
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    def forward(self, input):
        xy = self.features(self.input_norm(input))
        angle = torch.atan2(xy[:,0] + 1e-8, xy[:,1]+1e-8);
        return angle
    
try:
    input_img_fname = sys.argv[1]
    output_fname = sys.argv[2]
except:
    print ("Wrong input format. Try python estimate_angles_from_hpatches_file.py img.png angles.txt weights_dir")
    sys.exit(1)
if len(sys.argv) > 3:
    weights_dir = sys.argv[3]
else:
    weights_dir = '../../benchmark-orientation/matlab/src/KeypointOrientations/GHHPoolingEF/prelearned/efsift-360'

model = YiNet()
model.import_weights(weights_dir)
model.eval()
if USE_CUDA:
    model = model.cuda()
    
image = cv2.imread(input_img_fname,0)
h,w = image.shape
n_patches =  int(h/w)

descriptors_for_net = np.zeros((n_patches, 1))
t = time.time()
patches = np.ndarray((n_patches, 1, PS, PS), dtype=np.float32)
for i in range(n_patches):
    patch =  image[i*(w): (i+1)*(w), 0:w]
    patches[i,0,:,:] = cv2.resize(patch,(PS,PS))
bs = 128;
outs = []
n_batches = int(n_patches / bs) + 1
t = time.time()

for batch_idx in range(n_batches):
    st = batch_idx * bs
    if batch_idx == n_batches - 1:
        if (batch_idx + 1) * bs > n_patches:
            end = n_patches
        else:
            end = (batch_idx + 1) * bs
    else:
        end = (batch_idx + 1) * bs
    if st >= end:
        continue
    data_a = patches[st:end, :, :, :].astype(np.float32)
    data_a = torch.from_numpy(data_a)
    if USE_CUDA:
        data_a = data_a.cuda()
    data_a = Variable(data_a, volatile=True)
    out_a = model(data_a)
    descriptors_for_net[batch_idx * bs: end,:] = out_a.data.cpu().numpy().reshape(-1, 1)
et  = time.time() - t
print ('processing', et, et/float(n_patches), ' per patch')
np.savetxt(output_fname, descriptors_for_net, delimiter=' ', fmt='%10.5f')    