# import global packages
import os
import sys
import pathlib
import shutil
import argparse
from re import A
from ast import arg
from termcolor import colored

# import local files
sys.path.append(os.path.join(sys.path[0], 'img-process'))
sys.path.append(os.path.join(sys.path[0], 'utils'))
sys.path.append(os.path.join(sys.path[0], 'segmentation/unet'))
import params_meta as pm
from unet_seg import UnetSeg
from data_prep import UnetPrep
# from eval_results import Evaluate
from gbmw_fpw import GBMW_FPW

# ============================== run-time functions ==============================
# initialize image preprocess
def setup_preprocess(verbose):
    Preprocess = UnetPrep(
        data_path = pm.PATH[pm.env],
        datasets = [pm.DATASET[d] for d in pm.datasets],
        sliding_step = pm.sliding_step,
        verbose = verbose
    )
    Preprocess.update_image_stats()
    return Preprocess

def preprocess_data(verbose):
    print(colored(('#'*25 + ' Preprocessing Dataset ' + '#'*25), 'green'))
    Preprocess = setup_preprocess(verbose)
    Preprocess.update_image_list(pm.kfold)
    Preprocess.generate_image_tiles(0,pm.kfold)

def k_fold(verbose):
    print(colored(('#'*25 + ' Start Training with k-fold Validation ' + '#'*25), 'green'))
    Preprocess = setup_preprocess(verbose)
    DeepSeg = init_model(verbose)
    for n in range(pm.kfold):
        save_dir = 'fold_%s/' % str(n+1)
        Preprocess.generate_image_tiles(n,pm.kfold)
        train_model(DeepSeg, save_dir)
    
def evaluate_results():
    for n in range(pm.kfold):
        Preprocess = setup_preprocess(verbose=False)
        Preprocess.generate_image_tiles(n,pm.kfold)
        CalcGBMW = GBMW_FPW()
        CalcGBMW.calc_manager('fold_%s' % str(n+1), 'unet_15_epochs')
    # CalcGBMW = GBMW_FPW()
    # CalcGBMW.calc_manager('fold_%s' % str(5), 'unet_15_epochs')



def init_model(verbose):
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
        criterion = pm.criterion,
    )
    DeepSeg.load_and_augment()
    return DeepSeg

def train_model(DeepSeg, save_dir):
    model_dir = pathlib.Path.cwd() / 'models' / save_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    DeepSeg.current_dir = save_dir
    DeepSeg.initialize_model()
    DeepSeg.train_model()


def init_and_train_model(verbose, current_fold=0):
    save_dir = ''
    DeepSeg = init_model(verbose)
    train_model(DeepSeg, save_dir)


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
    DeepSeg.initialize_model()
    Preprocess = setup_preprocess(verbose)
    abs_path = pathlib.Path.cwd() / 'pred'
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

    for n in range(pm.kfold):
        Preprocess.generate_image_tiles(n,pm.kfold)
        cur_fold = 'fold_%s/' % str(n+1)
        for m in pm.models:
            m_name = cur_fold + 'unet_' + m + '_epochs'
            DeepSeg.load_and_predict(m_name, pm.out_channels)

def ask_if_proceed(callback, arg):
    val = input(colored("This command will delete previously trained models, are you sure to proceed? [y/n]: ", 'yellow'))
    if val == 'y':
        if (arg is not None): callback(arg)
        else: callback()
    elif val != 'n':
        print(colored('invalid input', 'red'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process train arguments')
    parser.add_argument('-prep', '--preprocess', action='store_true', help='preprocess dataset')
    parser.add_argument('-train', '--init_train', action='store_true', help='train and initialize model')
    parser.add_argument('-kfold', '--kfold', action='store_true', help='train using kfold cross-validation')
    # parser.add_argument('-c', '--cont_train', action='store_true', help='continue training')
    # parser.add_argument('--integers', type=str, help='epoch num to continue training')
    parser.add_argument('-pred', '--predict', action='store_true', help='predict output')
    parser.add_argument('-v', '--verbose', action='store_true', help='turn on verbose')
    parser.add_argument('-eval', '--evaluation', action='store_true', help='evaluate the model')
    args = parser.parse_args()

    if args.preprocess:
        preprocess_data(args.verbose)
    elif args.kfold:
        callback = k_fold
        arg = args.verbose
        ask_if_proceed(callback, arg)
    elif args.init_train:
        callback = init_and_train_model
        arg = args.verbose
        ask_if_proceed(callback, arg)
    elif args.predict:
        predict_results(args.verbose)
    elif args.evaluation:
        evaluate_results()
    # elif args.cont_train:
    #   print(args.integers)
    else:
        print(colored('please input arg(s)', 'red'))
