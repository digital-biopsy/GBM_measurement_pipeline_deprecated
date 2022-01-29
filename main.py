# import global packages
import sys, os
sys.path.append(os.path.join(sys.path[0], 'data-preprocess'))
# import local files
from image_prep import ImgPrep

# initialize image preprocess
ImgPrep = ImgPrep()
ImgPrep.update_image_list()
ImgPrep.generate_image_tiles()