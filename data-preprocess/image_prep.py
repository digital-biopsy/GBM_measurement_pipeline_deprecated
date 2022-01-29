import os
import sys
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

class ImgPrep:

  # initialize class
  def __init__(self):
    # parameters
    self.sliding_step = 256
    self.crop_size = 512
    self.shrink_factor = 0.5

    # change to your local data path (where raw image/labels are stored)
    self.data_path = '/Users/ericwang/Desktop/Research/Digital Biopsy/train-data/'
    self.train_dir = 'data'


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
    path_list = ['inputs', 'labels', 'test/inputs', 'test/labels', 'val/inputs', 'val/labels']
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
      if (file_name + '-GBMlabels' != label_name):
        print('train test sets don\'t match at ', callback)
        print(file_name, label_name)
        return False
    return True


  def crop(self, image, label, count, image_path, label_path):
    # find image shape
    shape = image.shape
    # calculate image/label number to make each file name identical
    tilecols = (shape[1] - self.crop_size) // self.sliding_step
    tilerows = (shape[0] - self.crop_size) // self.sliding_step
    start_idx = tilerows * tilecols * count
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
        # add binary threshold to the label
        (thresh, lb_crop) = cv2.threshold(lb_crop, 127, 255, cv2.THRESH_BINARY)
        # set save index
        save_idx = str(r * tilecols + c + 1 + start_idx) + '.png'
        cv2.imwrite(image_path + '/' + save_idx, img_crop)
        cv2.imwrite(label_path + '/' + save_idx, lb_crop)


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
          input_dir = os.path.join(sys.path[0], 'data', input_dir)
          label_dir = os.path.join(sys.path[0], 'data', label_dir)
          self.crop(raw, label, i, input_dir, label_dir)
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
    self.crop_image_label(x_train, y_train, 'inputs', 'labels',)
    self.crop_image_label(x_test, y_test, 'test/inputs', 'test/labels',)


  def update_image_list(self):
    # generate image and label file lists
    self.get_image_list('labels')
    self.get_image_list('inputs')