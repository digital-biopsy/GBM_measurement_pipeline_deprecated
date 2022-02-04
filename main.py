# import global packages
import sys, os
sys.path.append(os.path.join(sys.path[0], 'data-preprocess'))
# import local files
from data_prep import UnetPrep

# initialize image preprocess
def preprocess_data():
  Preprocess = UnetPrep()
  Preprocess.sliding_step = 256
  # Preprocess.seg_type = '-SDDlabels' # change to -GBMlabel when cropping GBMs
  Preprocess.data_path = '/Users/ericwang/Desktop/Research/Digital Biopsy/train-data/' # change to train-data when cropping GBMs
  Preprocess.update_image_list()
  Preprocess.generate_image_tiles()

preprocess_data()