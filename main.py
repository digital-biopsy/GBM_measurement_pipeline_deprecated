# import global packages
import os
import sys
import torch
import argparse
sys.path.append(os.path.join(sys.path[0], 'data-preprocess'))
sys.path.append(os.path.join(sys.path[0], 'evaluation'))
sys.path.append(os.path.join(sys.path[0], 'segmentation', 'unet'))
from re import A
from ast import arg
# import local files
from data_prep import UnetPrep
from eval_results import Evaluate
from segmentation import Segmentation

ENV = {
  'server': '/hy-tmp/',
  'local': '/Users/ericwang/Desktop/Research/Digital-Biopsy/'
}
DATASET = {
  '4wks': 'train-data-4wks/',
  '16wks': 'train-data-16wks/'
}
DEVICE = {
  'server': 'cuda',
  'local': 'cpu'
}
dataset = '16wks'
env = 'server'

# initialize image preprocess
def preprocess_data(verbose):
  Preprocess = UnetPrep(verbose)
  Preprocess.sliding_step = 256
  # Preprocess.seg_type = '-SDDlabels' # change to -GBMlabel when cropping GBMs
  Preprocess.data_path = ENV[env] + DATASET[dataset] # change to train-data when cropping GBMs
  Preprocess.update_image_stats()
  Preprocess.update_image_list()
  Preprocess.generate_image_tiles()

def evaluate_results():
  Eval = Evaluate()
  Eval.evaluate()

def init_and_train_model(verbose):
  data_path = 'data'
  nums_epochs = 40
  cross_entropy_weight = 0.045
  fit_steps = 500
  device = DEVICE[env]

  global DeepSeg
  DeepSeg = Segmentation(
    data_path=data_path,
    epochs=nums_epochs,
    weight=cross_entropy_weight,
    fit_steps=fit_steps,
    device=device,
    verbose=verbose,
  )
  DeepSeg.load_and_augment()
  DeepSeg.initialize_model()
  DeepSeg.train_model()
  DeepSeg.load_val_images()

# def train_model(verbose):
#   print('train_model')
#   if 'DeepSeg' in globals():
#     if DeepSeg.init:
#       print('start training')
#       DeepSeg.train_model()
#       DeepSeg.load_val_images()
#     else:
#       print('Model not initialized, reinitializing ...')
#       init_and_train_model(verbose)
#   else:
#     print('DeepSeg not initialized, reinitializing ...')
#     init_and_train_model(verbose)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process train arguments')
  parser.add_argument('-p', '--preprocess', action='store_true', help='preprocess data')
  parser.add_argument('-t', '--init_train', action='store_true', help='train and initialize model')
  # parser.add_argument('-t', '--train', action='store_true', help='train model')
  parser.add_argument('-v', '--verbose', action='store_true', help='turn on verbose')
  parser.add_argument('-e', '--evaluation', action='store_true', help='evaluate the model')
  args = parser.parse_args()

  if args.preprocess:
    preprocess_data(args.verbose)
  elif args.init_train:
    init_and_train_model(args.verbose)
  # elif args.train:
  #   train_model(args.verbose)
  elif args.evaluation:
    evaluate_results()
  else:
    print('please input arg(s)')
