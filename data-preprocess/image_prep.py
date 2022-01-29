import os
import sys
import glob

class ImgPrep:
  # initialize class
  def __init__(self):
    # parameters
    self.sliding_step = 128
    self.crop_size = 512
    self.shrink_factor = 0.5

    # change to your local data path (where raw image/annotations are stored)
    self.data_path = '/Users/ericwang/Desktop/Research/Digital Biopsy/train-data/'
    self.train_dir = '/data'

  def get_image_list(self, sub_dir):
    # save image list to 
    data_path = os.path.join(self.data_path, sub_dir)

    file_name = sys.path[0] + self.train_dir + '/' + sub_dir + '.txt'
    file = open(file_name, 'w')

    for root, _, files in os.walk(data_path):
      for name in files:
        if name != '.DS_Store':
          file.write(os.path.join(root, name) + '\n')
  
    file.close()

  def generate_image_tiles(self):
    self.get_image_list('annotations')
    self.get_image_list('images')

