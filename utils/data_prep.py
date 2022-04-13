import os
import sys
import cv2
import shutil
import pathlib
import pandas as pd
import numpy as np
from random import shuffle
from itertools import compress
from termcolor import colored
from sklearn.model_selection import train_test_split

class UnetPrep:

    # initialize class
    def __init__(self, data_path, datasets, sliding_step, verbose=False):
        # parameters
        self.verbose = verbose
        self.sliding_step = sliding_step
        self.crop_size = 512
        self.shrink_factor = 0.5

        # change to your local data path (where raw image/labels are stored)
        self.seg_type = '-GBMlabels'
        self.data_path = data_path
        self.datasets = datasets
        self.train_dir = 'data'

        # import image info and save to tile info
        self.img_stats = pd.DataFrame()
        self.tile_stats = pd.DataFrame(columns=['image', 'tile_index', 'input_directory', 'label_directory', 'gbmw', 'fpw', 'sdd', 'gbml'])


    def update_image_stats(self):
        # update image stats (read csv) after data_path is defined
        self.img_stats = pd.read_csv(os.path.join(self.data_path + self.datasets[0], 'stats.csv'))

    def __get_image_list__(self, sub_dir):
        """
        Get image list without using kfold.
        """
        file_name = os.path.join(sys.path[0], self.train_dir, sub_dir + '.txt')
        if os.path.exists(file_name): 
            os.remove(file_name)
        file = open(file_name, 'w')

        for dataset in self.datasets:
            data_path = os.path.join(self.data_path + dataset, sub_dir)

            for root, _, files in os.walk(data_path):
                files = sorted(files)
                for name in files:
                    if name != '.DS_Store':
                        file.write(os.path.join(root, name) + '\n')
              # file.write(os.path.join(root, files[-1])) # avoid write a blank line
        remove_chars = len(os.linesep)
        file.truncate(file.tell() - remove_chars)
        file.close()
    
    def __get_kfold_image_list__(self, root_path, sub_dir, sub_name):
        """
        Get images list in kfold fashion
        """
        # save image list path
        data_path = os.path.join(root_path, sub_dir)
        file_name = sub_dir + '_' + sub_name + '.txt'
        file_dir = os.path.join(sys.path[0], self.train_dir, 'kfold')
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        file = pathlib.Path(os.path.join(file_dir, file_name))
        file.touch(exist_ok=True)
        file = open(file, 'w')

        for root, _, files in os.walk(data_path):
            files = sorted(files)
            for name in files[:-1]:
                if name != '.DS_Store':
                    file.write(os.path.join(root, name) + '\n')
            file.write(os.path.join(root, files[-1])) # avoid writing a blank line at the end of the file
        file.close()
    

    def __read_image_list__(self, input_dir, label_dir, k):
        """
        Read dataset and return input list, label list, tar-image 
        correspondance list, and target (animal) list.
        """
        input_list, label_list, tar_list, tar_unique = [], [], [], []
        for sub in self.datasets:
            sub_name = sub.split('-')[-1].split('/')[0]
            ipt = self.__read_files_list__(input_dir+'_'+sub_name)
            lbl = self.__read_files_list__(label_dir+'_'+sub_name)

            input_list.append(ipt)
            label_list.append(lbl)
            
            target_idx = [i.split('/')[-1].split('-')[0] for i in ipt]
            tar_list.append(target_idx)

            tar_unq = list(set(target_idx))
            shuffle_idx = list(range(k))
            shuffle(shuffle_idx)
            tar_unique.append([tar_unq[shuffle_idx[n]] for n in range(k)])
        
        return [input_list, label_list, tar_list, tar_unique]


    def __shuffle_image_list__(self, k):
        """
        Shuffle image list with respect to its corresponding animal. i.e. images belond to
        the same animal will stay together after being shuffled.
        """
        input_dir = 'kfold/inputs'
        label_dir = 'kfold/labels'
        [input_list, label_list, tar_list, tar_unique] = self.__read_image_list__(input_dir, label_dir, k)

        # clear dir
        kfold_dir = pathlib.Path.cwd() / 'data' / 'kfold'
        if not os.path.exists(kfold_dir): os.makedirs(kfold_dir)

        for n in range(k):
            save_dir = 'fold_%s/' % str(n+1)
            save_dir = kfold_dir / save_dir
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            
            inputs, labels = [], []
            for i in range(len(self.datasets)):
                cur_tar = tar_unique[i]
                filter = [tar_list[i][j]==cur_tar[n] for j in range(len(tar_list[i]))]
                ipt = list(compress(input_list[i], filter))
                lbl = list(compress(label_list[i], filter))
                inputs += ipt
                labels += lbl

            self.__save_image_data__(save_dir, inputs, labels)

    
    def __save_image_data__(self, save_dir, inputs, labels):
        """Save image list into the inputs.txt and labels.txt in its correspoding kfold folder"""
        input_file = pathlib.Path(os.path.join(sys.path[0], save_dir, 'inputs.txt'))
        label_file = pathlib.Path(os.path.join(sys.path[0], save_dir, 'labels.txt'))
        input_file.touch(exist_ok=True)
        label_file.touch(exist_ok=True)
        input_file = open(input_file, 'w')
        label_file = open(label_file, 'w')

        for i in range(len(inputs)-1):
            input_file.write(inputs[i] + '\n')
            label_file.write(labels[i] + '\n')
        input_file.write(inputs[-1]) # avoid writing a blank line at the end of the file
        label_file.write(labels[-1]) # avoid writing a blank line at the end of the file
        input_file.close()
        label_file.close()


    def __make_data_dirs__(self):
        """
        Sets up data directories
        """
        path_list = ['inputs', 'labels', 'test/inputs', 'test/labels']
        for path in path_list:
            abs_path = os.path.join(sys.path[0], self.train_dir, path)
            if os.path.exists(abs_path):
                shutil.rmtree(abs_path)
                os.makedirs(abs_path)
            else:
                os.makedirs(abs_path)


    def __check_matches__(self, input_list, label_list, callback):
        """
        Check if image and label name are matched before cropping them
        """
        for i in range(len(input_list)):
            file_name = input_list[i].split('inputs/')[1].split('.jpg')[0]
            label_name = label_list[i].split('labels/')[1].split('.png')[0]
            if (file_name + self.seg_type != label_name):
                print(colored(('train test sets don\'t match at ' + callback), 'red'))
                print(colored((file_name + label_name), 'red'))
                return False
        return True


    def __crop__(self, image, label, count, image_path, label_path, start_idx):
        """
        Crop and save image/label tile
        """
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
        """
        Cropping manager, match the image and its corresponding label before feeding them into the cropping function
        """
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
        """
        Read image files from the image paths previously stored in the txt files
        """
        file_name = pathlib.Path(os.path.join(sys.path[0], self.train_dir, sub_dir + '.txt'))
        file_name.touch(exist_ok=True)
        file = open(file_name, "r")
        data_list = file.read().split("\n")
        file.close()
        return data_list

    
    # methods
    def generate_image_tiles(self, n=0, k=5):
        """
        Process images into image tiles
        """
        print(colored(('#'*25 + ' Train-test Split ' + '#'*25), 'green'))
        # reset tile_stats
        self.tile_stats = pd.DataFrame(columns=['image', 'tile_index', 'input_directory', 'label_directory', 'gbmw', 'fpw', 'sdd', 'gbml'])
        # create training directories
        self.__make_data_dirs__()

        x_train, y_train, x_test, y_test = [], [], [], []
        for fold in range(k):
            # read files into arrays
            fold_dir = 'kfold/fold_%s/' % str(fold+1)
            inputs = self.__read_files_list__(fold_dir + 'inputs')
            labels = self.__read_files_list__(fold_dir + 'labels')
            self.__check_matches__(inputs, labels, 'files read')

            if fold == n:
                x_test += inputs
                y_test += labels
            else:
                x_train += inputs
                y_train += labels
      
        # crop images
        self.__crop_image_label__(x_train, y_train, 'inputs', 'labels')
        self.__crop_image_label__(x_test, y_test, 'test/inputs', 'test/labels')

        csv_path = os.path.join(sys.path[0], self.train_dir, 'tile_stats.csv')
        if os.path.isfile(csv_path): os.remove(csv_path)
        self.tile_stats.to_csv(csv_path)


    def update_image_list(self, k):
      # generate image and label file lists
      for d in self.datasets:
          root_path = self.data_path + d
          sub_name = d.split('-')[-1].split('/')[0]
          self.__get_kfold_image_list__(root_path, 'labels', sub_name)
          self.__get_kfold_image_list__(root_path, 'inputs', sub_name)
      self.__shuffle_image_list__(k)