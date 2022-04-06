import os
import sys
import torch
from termcolor import colored
sys.path.append(os.path.join(sys.path[0], 'segmentation/unet'))
from unet_loss import IoULoss, FocalLoss, DiceLoss, TverskyLoss

# ============================== parameters ==============================
# global parameters
env = 'server'
DEVICE = {
  'server': 'cuda',
  'local': 'cpu'
}

# pre-processing parameters
datasets = ['ILK', '4wks']
kfold = 5 # 0 if all in inputs, 1 if all in tests, 5 by default (80% train 20% test)
sliding_step = 300
PATH = {
  'server': '/hy-tmp/',
  'local': '/Users/ericwang/Desktop/Research/Digital-Biopsy/'
}
DATASET = {
  '4wks': 'train-data-4wks/',
  '16wks': 'train-data-16wks/',
  'ILK': 'train-data-ILK/'
}

# model parameters
nums_epochs = 15
fit_steps = 500
channel_dims = 1
out_channels = 1
device = DEVICE[env]
batch_size = 2
start_filters = 64

# loss functions
loss_func = 'WCE'
cross_entropy_weight = 0.095

if loss_func == 'WCE':
  out_channels = 2
  weights = [1, cross_entropy_weight]
  class_weights = torch.FloatTensor(weights).cuda()
  criterion = torch.nn.CrossEntropyLoss(weight=class_weights) #CrossEntropyLoss
elif loss_func == 'IoU':
  criterion = IoULoss()
elif loss_func == 'Dice':
  criterion = DiceLoss()
elif loss_func == 'Focal':
  criterion = FocalLoss()
elif loss_func == 'Tversky':
  criterion = TverskyLoss()
else:
  print(colored('please use a valid loss function', 'red'))

# prediction parameters
models = ['45']