import os
import sys
import cv2
import shutil
import pandas as pd
import numpy as np
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
    self.train_dir = 'data'

    # import image info and save to tile info
    self.img_stats = pd.DataFrame()
    self.tile_stats = pd.DataFrame(columns=['image', 'tile_index', 'input_directory', 'label_directory', 'gbmw', 'fpw', 'sdd', 'gbml'])

  def update_image_stats(self):
    # update image stats (read csv) after data_path is defined
    self.img_stats = pd.read_csv(os.path.join(self.data_path, 'stats.csv'))

  def get_image_list(self, sub_dir):
    # save image list path
    data_path = os.path.join(self.data_path, sub_dir)

    file_name = os.path.join(sys.path[0], self.train_dir, sub_dir + '.txt')
    file = open(file_name, 'w')

    for root, _, files in os.walk(data_path):
      files = sorted(files)
      for name in files[:-1]:
        if name != '.DS_Store':
          file.write(os.path.join(root, name) + '\n')
      file.write(os.path.join(root, files[-1])) # avoid write a blank line
    file.close()
  

  def make_train_dirs(self):
    path_list = ['inputs', 'labels', 'test/inputs', 'test/labels']
    for path in path_list:
      abs_path = os.path.join(sys.path[0], self.train_dir, path)
      if os.path.exists(abs_path):
        shutil.rmtree(abs_path)
        os.makedirs(abs_path)
      else:
        os.makedirs(abs_path)


  def check_matches(self, input_list, label_list, callback):
    for i in range(len(input_list)):
      file_name = input_list[i].split('inputs/')[1].split('.jpg')[0]
      label_name = label_list[i].split('labels/')[1].split('.png')[0]
      if (file_name + self.seg_type != label_name):
        print('train test sets don\'t match at ', callback)
        print(file_name, label_name)
        return False
    return True


  def crop(self, image, label, count, image_path, label_path):
    # find image shape
    shape = image.shape
    # calculate image/label number to make each file name identical
    tilecols = ((shape[1] - self.crop_size) // self.sliding_step) + 1
    tilerows = ((shape[0] - self.crop_size) // self.sliding_step) + 1
    start_idx = tilerows * tilecols * count
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
    return idx_list


  def save_info(self, idx_list, img_path, input_dir, label_dir):
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


  def crop_image_label(self, inputs, labels, input_dir, label_dir):
    if self.check_matches(inputs, labels, 'files crop'):
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
          idx_list = self.crop(raw, label, i, abs_input_dir, abs_label_dir)
          self.save_info(idx_list, inputs[i], input_dir, label_dir)
        else:
          print('image size and label size does not match')


  def read_files_list(self, sub_dir):
    # read image list path
    file_name = os.path.join(sys.path[0], self.train_dir, sub_dir + '.txt')
    file = open(file_name, "r")
    data_list = file.read().split("\n")
    file.close()
    return data_list

  
  # methods
  def generate_image_tiles(self, k_fold = False):
    # create training directories
    self.make_train_dirs()

    # read files into arrays
    inputs = self.read_files_list('inputs')
    labels = self.read_files_list('labels')
    self.check_matches(inputs, labels, 'files read')

    # random split dataset
    x_train, x_test, y_train, y_test = train_test_split(
      inputs, labels, test_size=0.2, random_state=42)

    # crop images
    self.crop_image_label(x_train, y_train, 'inputs', 'labels')
    self.crop_image_label(x_test, y_test, 'test/inputs', 'test/labels')

    csv_path = os.path.join(sys.path[0], self.train_dir, 'tile_stats.csv')
    self.tile_stats.to_csv(csv_path)


  def update_image_list(self):
    # generate image and label file lists
    self.get_image_list('labels')
    self.get_image_list('inputs')