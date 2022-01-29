import os
import glob

class ImgPrep:
  # initialize class
  def __init__(self):
    # parameters
    self.sliding_step = 128
    self.crop_size = 512
    self.shrink_factor = 0.5

    self.data_path = '/Users/ericwang/Desktop/Research/Digital Biopsy/train-data'
  
  def get_original_files(self):
    for root, dirs, files in os.walk(self.data_path):
      for name in files:
          print(os.path.join(root, name))
      for name in dirs:
          print(os.path.join(root, name))

  def generate_image_tiles(self):
    self.get_original_files()

