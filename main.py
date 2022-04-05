# import global packages
import os
import sys
import pathlib
import shutil
import argparse
from re import A
from ast import arg

# import local files
sys.path.append(os.path.join(sys.path[0], 'data-preprocess'))
sys.path.append(os.path.join(sys.path[0], 'evaluation'))
sys.path.append(os.path.join(sys.path[0], 'segmentation/unet'))
import params_meta as pm
from unet_seg import UnetSeg
from data_prep import UnetPrep
from eval_results import Evaluate

# ============================== run-time functions ==============================
# initialize image preprocess
def preprocess_data(verbose, task):
  print('#'*25, ' preprocessing dataset using %s mode ' % task, '#'*25)
  if task == 'shuffle':
    Preprocess = UnetPrep(verbose, pm.split_ratio)
    Preprocess.sliding_step = pm.sliding_step
    Preprocess.data_path = pm.PATH[pm.env]
    Preprocess.datasets = [pm.DATASET[d] for d in pm.datasets]
    Preprocess.update_image_stats()
    Preprocess.update_image_list()
    Preprocess.generate_image_tiles()
  elif task == '5fold':
    Preprocess = UnetPrep(verbose, pm.split_ratio)
    Preprocess.sliding_step = pm.sliding_step
    Preprocess.data_path = pm.PATH[pm.env]
    Preprocess.datasets = [pm.DATASET[d] for d in pm.datasets]
    Preprocess.update_image_stats()
    Preprocess.update_image_list()
    Preprocess.generate_image_tiles()
    
def evaluate_results():
  Eval = Evaluate()
  Eval.evaluate()

def init_and_train_model(verbose):
  DeepSeg = UnetSeg(
    data_path = 'data',
    epochs = pm.nums_epochs,
    weight = pm.cross_entropy_weight,
    fit_steps = pm.fit_steps,
    device = pm.device,
    out_channels = pm.out_channels,
    batch_size = pm.batch_size,
    loss_func = pm.loss_func,
    channel_dims = pm.channel_dims,
    verbose = verbose,
    start_filters = pm.start_filters,
    criterion = pm.criterion
  )
  model_dir = pathlib.Path.cwd() / 'models'
  if not os.path.exists(model_dir):
      os.makedirs(model_dir)
  DeepSeg.load_and_augment()
  DeepSeg.train_model()

def predict_results(verbose):
  DeepSeg = UnetSeg(
    data_path ='data',
    epochs = pm.nums_epochs,
    weight = pm.cross_entropy_weight,
    fit_steps = pm.fit_steps,
    device = pm.device,
    out_channels = pm.out_channels,
    batch_size = pm.batch_size,
    loss_func = pm.loss_func,
    channel_dims = pm.channel_dims,
    verbose = verbose,
    start_filters = pm.start_filters,
    criterion = pm.criterion
  )
  abs_path = pathlib.Path.cwd() / 'data' / 'pred'
  if not os.path.exists(abs_path):
      os.makedirs(abs_path)
  for m in pm.models:
    m_name = 'unet_' + m + '_epochs'
    DeepSeg.load_and_predict(m_name, pm.out_channels)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process train arguments')
  parser.add_argument('-prep', '--preprocess', action='store_true', help='preprocess dataset')
  parser.add_argument('-prep_opt', '--prep_opt', default='shuffle', help='shuffle | 5fold')
  parser.add_argument('-train', '--init_train', action='store_true', help='train and initialize model')
  # parser.add_argument('-c', '--cont_train', action='store_true', help='continue training')
  # parser.add_argument('--integers', type=str, help='epoch num to continue training')
  parser.add_argument('-pred', '--predict', action='store_true', help='predict output')
  parser.add_argument('-v', '--verbose', action='store_true', help='turn on verbose')
  parser.add_argument('-eval', '--evaluation', action='store_true', help='evaluate the model')
  args = parser.parse_args()

  assert (args.prep_opt in ['shuffle', '5fold']), 'preproceesing should be shuffle or 5fold'

  if args.preprocess:
    preprocess_data(args.verbose, args.prep_opt)
  elif args.init_train:
    val = input("This might delete previously trained models, are you sure to proceed? [y/n]: ")
    if val == 'y':
      init_and_train_model(args.verbose)
    elif val != 'n':
      print('invalid input')
  elif args.predict:
    predict_results(args.verbose)
  elif args.evaluation:
    evaluate_results()
  # elif args.cont_train:
  #   print(args.integers)
  else:
    print('please input arg(s)')
