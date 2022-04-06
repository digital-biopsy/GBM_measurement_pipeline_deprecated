import os
import sys
import cv2
import shutil
import pandas as pd
import numpy as np
from random import shuffle
from termcolor import colored
from sklearn.model_selection import train_test_split

class UnetPrep:

  # initialize class
  def __init__(self, verbose=False):
    # parameters
    self.verbose = verbose
    self.sliding_step = 256
    self.crop_size = 512
    self.shrink_factor = 0.5

    # change to your local data path (where raw image/labels are stored)
    self.seg_type = '-GBMlabels'
    self.data_path = ''
    self.datasets = []
    self.train_dir = 'data'

    # import image info and save to tile info
    self.img_stats = pd.DataFrame()
    self.tile_stats = pd.DataFrame(columns=['image', 'tile_index', 'input_directory', 'label_directory', 'gbmw', 'fpw', 'sdd', 'gbml'])


  def update_image_stats(self):
    # update image stats (read csv) after data_path is defined
    self.img_stats = pd.read_csv(os.path.join(self.data_path + self.datasets[0], 'stats.csv'))


  def __get_image_list__(self, root_path, sub_dir):
    # save image list path
    data_path = os.path.join(root_path, sub_dir)

    file_name = os.path.join(sys.path[0], self.train_dir, sub_dir + '.txt')
    file = open(file_name, 'w')

    for root, _, files in os.walk(data_path):
      files = sorted(files)
      for name in files[:-1]:
        if name != '.DS_Store':
          file.write(os.path.join(root, name) + '\n')
      file.write(os.path.join(root, files[-1])) # avoid writing a blank line at the end of the file
    file.close()
  

  def __shuffle_image_list__(self, input_dir, label_dir):
    inputs = self.__read_files_list__(input_dir)
    labels = self.__read_files_list__(label_dir)

    idx_list = list(range(len(inputs)))
    shuffle(idx_list)

    input_file_name = os.path.join(sys.path[0], self.train_dir, input_dir + '.txt')
    label_file_name = os.path.join(sys.path[0], self.train_dir, label_dir + '.txt')
    input_file = open(input_file_name, 'w')
    label_file = open(label_file_name, 'w')

    for i in idx_list[:-1]:
      input_file.write(inputs[i] + '\n')
      label_file.write(labels[i] + '\n')

    input_file.write(inputs[-1]) # avoid writing a blank line at the end of the file
    label_file.write(labels[-1]) # avoid writing a blank line at the end of the file
    input_file.close()
    label_file.close()
  

  def __make_train_dirs__(self):
    path_list = ['inputs', 'labels', 'test/inputs', 'test/labels']
    for path in path_list:
      abs_path = os.path.join(sys.path[0], self.train_dir, path)
      if os.path.exists(abs_path):
        shutil.rmtree(abs_path)
        os.makedirs(abs_path)
      else:
        os.makedirs(abs_path)


  def __check_matches__(self, input_list, label_list, callback):
    for i in range(len(input_list)):
      file_name = input_list[i].split('inputs/')[1].split('.jpg')[0]
      label_name = label_list[i].split('labels/')[1].split('.png')[0]
      if (file_name + self.seg_type != label_name):
        print(colored(('train test sets don\'t match at ' + callback), 'red'))
        print(colored((file_name + label_name), 'red'))
        return False
    return True


  def __crop__(self, image, label, count, image_path, label_path, start_idx):
    # find image shape
    shape = image.shape
    # calculate image/label number to make each file name identical
    tilecols = ((shape[1] - self.crop_size) // self.sliding_step) + 1
    tilerows = ((shape[0] - self.crop_size) // self.sliding_step) + 1
    start_idx
    index = 1
    idx_list = []
    # iterate over columns and rows of image
    for r in range(tilerows):
      row_idx = r * self.sliding_step
      for c in range(tilecols):
        col_idx = c * self.sliding_step
        lb_crop = label[
          row_idx:(row_idx + self.crop_size), 
          col_idx:(col_idx + self.crop_size)]
        img_crop = image[
          row_idx:(row_idx + self.crop_size), 
          col_idx:(col_idx + self.crop_size)]
        
        contains_gbm = abs((lb_crop-255).sum()) > 3000
        if contains_gbm:
          # add binary threshold to the label
          (thresh, lb_crop) = cv2.threshold(lb_crop, 127, 255, cv2.THRESH_BINARY)
          # set save index
          save_idx = str(index + start_idx) + '.png'
          cv2.imwrite(image_path + '/' + save_idx, img_crop)
          cv2.imwrite(label_path + '/' + save_idx, lb_crop)
          idx_list.append(save_idx)
          index += 1
    return idx_list, (index + start_idx - 1)


  def __save_info__(self, idx_list, img_path, input_dir, label_dir):
    tile_rows = []
    img_name = img_path.split('inputs/')[1]
    tile_row = self.img_stats.loc[self.img_stats['Image'] == img_name].values

    if len(tile_row) == 1:
      gbmw = tile_row[0][4]
      fpw = tile_row[0][5]
      sdd = tile_row[0][6]
      gbml = tile_row[0][7]/8.8779
    else:
      gbmw = np.nan
      fpw = np.nan
      sdd = np.nan
      gbml = np.nan

    for idx in idx_list:
      tile_rows.append([img_name, idx, input_dir, label_dir, gbmw, fpw, sdd, gbml])
    tile_block = pd.DataFrame(tile_rows, columns=['image', 'tile_index', 'input_directory', 'label_directory', 'gbmw', 'fpw', 'sdd', 'gbml'])
    self.tile_stats = pd.concat([self.tile_stats, tile_block], ignore_index=True)
    # sestats.append({'image': img_name, 'tile index': idx}, ignore_index=True)


  def __crop_image_label__(self, inputs, labels, input_dir, label_dir):
    if self.__check_matches__(inputs, labels, 'files crop'):
      start_index = 0
      for i in range(len(inputs)):
        # read images/labels in gray scale
        raw = cv2.imread(inputs[i], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(labels[i], cv2.IMREAD_GRAYSCALE)
        reshape = (int(raw.shape[1]*self.shrink_factor), int(raw.shape[0]*self.shrink_factor))

        # shrink image resolutions
        raw = np.array(cv2.resize(raw, reshape))
        label = np.array(cv2.resize(label, reshape))
        if raw.shape == label.shape:
          abs_input_dir = os.path.join(sys.path[0], self.train_dir, input_dir)
          abs_label_dir = os.path.join(sys.path[0], self.train_dir, label_dir)
          idx_list, start_index = self.__crop__(raw, label, i, abs_input_dir, abs_label_dir, start_index)
          self.__save_info__(idx_list, inputs[i], input_dir, label_dir)
        else:
          print(colored('image size and label size does not match', 'red'))


  def __read_files_list__(self, sub_dir):
    # read image list path
    file_name = os.path.join(sys.path[0], self.train_dir, sub_dir + '.txt')
    file = open(file_name, "r")
    data_list = file.read().split("\n")
    file.close()
    return data_list

  
  # methods
  def generate_image_tiles(self, n=0, k=5):
    # create training directories
    self.__make_train_dirs__()

    # read files into arrays
    inputs = self.__read_files_list__('inputs')
    labels = self.__read_files_list__('labels')
    self.__check_matches__(inputs, labels, 'files read')

    # random split dataset
    if k == 0:
      x_train = inputs
      y_train = labels
      x_test, y_test = [], []
    else:
      group_len = len(inputs) // k
      idx_start = n*group_len
      idx_end = (n+1)*group_len
      x_train = inputs[0:idx_start] + inputs[idx_end:]
      x_test = inputs[idx_start:idx_end]
      y_train = labels[0:idx_start] + labels[idx_end:]
      y_test = labels[idx_start:idx_end]

    # crop images
    self.__crop_image_label__(x_train, y_train, 'inputs', 'labels')
    self.__crop_image_label__(x_test, y_test, 'test/inputs', 'test/labels')

    csv_path = os.path.join(sys.path[0], self.train_dir, 'tile_stats.csv')
    os.remove(csv_path)
    self.tile_stats.to_csv(csv_path)


  def update_image_list(self):
    # generate image and label file lists
    for d in self.datasets:
      root_path = self.data_path + d
      self.__get_image_list__(root_path, 'labels')
      self.__get_image_list__(root_path, 'inputs')
    # self.__shuffle_image_list__('labels', 'inputs')