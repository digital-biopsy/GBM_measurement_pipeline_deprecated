# import global packages
import os
import sys
import torch
import pathlib
import shutil
import argparse
sys.path.append(os.path.join(sys.path[0], 'data-preprocess'))
sys.path.append(os.path.join(sys.path[0], 'evaluation'))
sys.path.append(os.path.join(sys.path[0], 'segmentation'))
from re import A
from ast import arg
# import local files
from loss import IoULoss
from loss import FocalLoss
from loss import DiceLoss
from loss import TverskyLoss
from data_prep import UnetPrep
from eval_results import Evaluate
from segmentation import Segmentation

# ============================== parameters ==============================
# global parameters
env = 'server'
DEVICE = {
  'server': 'cuda',
  'local': 'cpu'
}

# pre-processing parameters
dataset = '16wks'
sliding_step = 256
PATH = {
  'server': '/hy-tmp/',
  'local': '/Users/ericwang/Desktop/Research/Digital-Biopsy/'
}
DATASET = {
  '4wks': 'train-data-4wks/',
  '16wks': 'train-data-16wks/'
}

# model parameters
nums_epochs = 90
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
  print('please use a valid loss function')

# prediction parameters
models = ['40', '45']

# ============================== run-time functions ==============================
# initialize image preprocess
def preprocess_data(verbose):
  Preprocess = UnetPrep(verbose)
  Preprocess.sliding_step = sliding_step
  Preprocess.data_path = PATH[env] + DATASET[dataset] # change to train-data when cropping GBMs
  Preprocess.update_image_stats()
  Preprocess.update_image_list()
  Preprocess.generate_image_tiles()

def evaluate_results():
  Eval = Evaluate()
  Eval.evaluate()

def init_and_train_model(verbose):
  DeepSeg = Segmentation(
    data_path='data',
    epochs=nums_epochs,
    weight=cross_entropy_weight,
    fit_steps=fit_steps,
    device=device,
    out_channels = out_channels,
    batch_size = batch_size,
    loss_func = loss_func,
    channel_dims = channel_dims,
    verbose=verbose,
    start_filters=start_filters,
    criterion=criterion
  )
  abs_path = pathlib.Path.cwd() / 'models'
  shutil.rmtree(abs_path)
  os.makedirs(abs_path)
  DeepSeg.load_and_augment()
  DeepSeg.train_model()

def predict_results(verbose):
  DeepSeg = Segmentation(
    data_path='data',
    epochs=nums_epochs,
    weight=cross_entropy_weight,
    fit_steps=fit_steps,
    device=device,
    out_channels = out_channels,
    batch_size = batch_size,
    loss_func = loss_func,
    channel_dims = channel_dims,
    verbose=verbose,
    start_filters=start_filters,
    criterion=criterion
  )
  abs_path = pathlib.Path.cwd() / 'pred'
  shutil.rmtree(abs_path)
  os.makedirs(abs_path)
  for m in models:
    m_name = 'unet_' + m + '_epochs'
    DeepSeg.load_and_predict(m_name, out_channels)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process train arguments')
  parser.add_argument('-prep', '--preprocess', action='store_true', help='preprocess data')
  parser.add_argument('-train', '--init_train', action='store_true', help='train and initialize model')
  # parser.add_argument('-c', '--cont_train', action='store_true', help='continue training')
  # parser.add_argument('--integers', type=str, help='epoch num to continue training')
  parser.add_argument('-pred', '--predict', action='store_true', help='predict output')
  parser.add_argument('-v', '--verbose', action='store_true', help='turn on verbose')
  parser.add_argument('-eval', '--evaluation', action='store_true', help='evaluate the model')
  args = parser.parse_args()

  if args.preprocess:
    preprocess_data(args.verbose)
  elif args.init_train:
    val = input("Train will delete weight vectors previously trained, are you sure to proceed? [y/n]: ")
    if val == 'y':
      init_and_train_model(args.verbose)
    elif val != 'n':
      print('invalid input')
  # elif args.cont_train:
  #   print(args.integers)
  elif args.predict:
    predict_results(args.verbose)
  elif args.evaluation:
    evaluate_results()
  else:
    print('please input arg(s)')
