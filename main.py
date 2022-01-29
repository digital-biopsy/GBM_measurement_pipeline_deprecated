# import global packages
import sys, os
sys.path.append(os.path.join(sys.path[0], 'data-preprocess/'))
# import local files
from image_prep import ImgPrep

# initialize image preprocess class
ImgPrep = ImgPrep()
ImgPrep.generate_image_tiles()