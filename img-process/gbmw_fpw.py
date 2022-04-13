from skimage.io import imread
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage import morphology
import cv2 as cv
from skimage import measure
from skimage.measure import label, regionprops
from skimage import data, util
from scipy import ndimage
from sklearn.mixture import GaussianMixture
import math
from scipy.signal import savgol_filter
import peakutils
from skimage.morphology import skeletonize
import os
import pandas as pd
from termcolor import colored


class GBMW_FPW():
    def __init__(self):
        self.FPW_blur_const = 9; #blur width
        self.GBMW_blur_const = 3; #blur width
        self.pixels_per_nm = 0.12 #convert to nm
        self.img_dir = pathlib.Path.cwd() / "data" / "test" / "inputs"
        self.pred_root = pathlib.Path.cwd() / "pred"

        src_dir = pathlib.Path.cwd() / "data"
        self.img_data = pd.read_csv(os.path.join(src_dir, 'tile_stats.csv'))

        self.fold_dir = ""
        self.model_dir = ""
    
    def get_filenames_of_path(self, path: pathlib.Path, ext: str = "*"):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return sorted(filenames)
    
    def get_images(self):
        pred_dir = self.pred_root / self.fold_dir / self.model_dir
        images_names = self.get_filenames_of_path(self.img_dir)
        predicted_names = self.get_filenames_of_path(pred_dir)
        # read images and store them in memory
        images = [imread(img_name) for img_name in images_names]
        predicted = [imread(tar_name) for tar_name in predicted_names]
        return images_names, predicted, images

    def blur_mask(self, predicted, blur_const):
        blur = filters.gaussian(predicted, blur_const)
        val = filters.threshold_otsu(blur)
        mk = np.array(blur > val, dtype = bool) 
        mask_remove = morphology.remove_small_objects(mk, min_size=5000, connectivity=1)
        mask = np.uint8(mask_remove)
        return mask

    def number_segments(self, masked):
        label_img = label(masked, connectivity=masked.ndim)
        props = regionprops(label_img)
        num_segments = len(props)
        return num_segments

    def separate_segments(self, masked, j):
        label_im, nb_labels = ndimage.label(masked)
        separate = np.empty((512,512),dtype='object')
        separate = [label_im == j]
        onesegmentmask = separate[0]
        return onesegmentmask

    def get_num_edges(self, onesegmentmask):
        edges = cv.Canny(onesegmentmask.astype(np.uint8),0,0.15)
        label_img = label(edges, connectivity=edges.ndim)
        props = regionprops(label_img)
        num_edges = len(props)
        return num_edges

    def get_edge_vals(self, onesegmentmask,k,orig):
        edges = cv.Canny(onesegmentmask.astype(np.uint8),0,0.15)
        label_img = label(edges, connectivity=edges.ndim)
        props = regionprops(label_img)
        xcoords = props[k].coords[:,0]
        ycoords = props[k].coords[:,1]
        edge_vals = np.empty((len(xcoords)), dtype='int_')
        for i in range(0,len(xcoords)):
            edge_vals[i] = orig[xcoords[i]][ycoords[i]]
        return edge_vals

    def get_num_fp_sd_pixels(self, edge_vals):
        gm = GaussianMixture(n_components=2, random_state=0).fit(edge_vals.reshape(-1,1))
        fpsdpred = gm.predict(edge_vals.reshape(-1,1))
        sd = sum(fpsdpred)
        fp = len(fpsdpred) - sd
        if gm.means_[0][0] > gm.means_[1][0]:
            sdmu = gm.means_[0][0]
            fpmu = gm.means_[1][0]
            sd_sig = math.sqrt(gm.covariances_[0][0][0])
            fp_sig = math.sqrt(gm.covariances_[1][0][0])
        else: 
            sdmu = gm.means_[1][0]
            fpmu = gm.means_[0][0]
            sd_sig = math.sqrt(gm.covariances_[1][0][0])
            fp_sig = math.sqrt(gm.covariances_[0][0][0])
        return fp, sd, fpmu, sdmu, fp_sig, sd_sig

    def find_sdd_fpw(self, edge_vals, sd_sig):
        edge_len = len(edge_vals)
        # window_len = edge_len//4 * 2 + 1
        if edge_len > 15:
            smoothvals = savgol_filter(edge_vals,15,5) ####
            indexes = peakutils.peak.indexes(smoothvals,thres=sd_sig/smoothvals.mean(),min_dist=15) ####
            numsd = len(indexes)
        else:
            print('Edge length under 15')
            numsd = 1
        if (numsd == 0): numsd = 1
        mfpw = len(edge_vals)/numsd #could subtract one from 
        return numsd, mfpw

    def find_area_gbmw(self, seg_mask):
        area = sum(sum(seg_mask))
        skeleton = skeletonize(seg_mask)
        length = sum(sum(skeleton))
        gbmw = area/length
        return area,gbmw

    def calc(self, cur_fold, cur_model):
        print(colored(('#'*25 + ' Calculating FPW and GBMW ' + '#'*25), 'green'))
        #Combined main script (ignore RunTimeWarnings)
        self.fold_dir = cur_fold
        self.model_dir = cur_model
        [images_names, predicted, original] = self.get_images()
        mask_FPW = np.empty((len(predicted),512,512), dtype='int_')
        mask_GBMW = np.empty((len(predicted),512,512), dtype='int_')
        gbmw_nm = np.empty((len(predicted)), dtype='object')
        total_area = np.empty((len(predicted)), dtype='object')
        fpw_nm = np.empty((len(predicted)), dtype='object')
        total_sd = np.empty((len(predicted)), dtype='object')
        for i in range(0,len(predicted)):
            predict = predicted[i]
            orig = original[i]
            
            masked_MFPW = self.blur_mask(predict, self.FPW_blur_const)
            masked_GBMW = self.blur_mask(predict, self.GBMW_blur_const)
            mask_FPW[i] = masked_MFPW #can remove, just to see masks
            mask_GBMW[i] = masked_GBMW
            num_segments = self.number_segments(masked_MFPW)
            tile_segs_num_sd = []
            tile_segs_mean_fpw = []
            tile_segs_area = []
            tile_segs_gbmw = []
            for j in range(1,num_segments+1):
                one_segment_mask_FPW = self.separate_segments(masked_MFPW,j)
                one_segment_mask_GBMW = self.separate_segments(masked_GBMW,j)
                num_edges = self.get_num_edges(one_segment_mask_FPW)
                if num_edges == 2:
                    [area,gbmw] = self.find_area_gbmw(one_segment_mask_GBMW)
                    tile_segs_area.append(area)
                    tile_segs_gbmw.append(gbmw)
                    indicate = np.empty((num_edges), dtype='float')
                    k = 0
                    edge_vals = self.get_edge_vals(one_segment_mask_FPW,k,orig)
                    [fp, sd, fpmu, sdmu, fp_sig, sd_sig] = self.get_num_fp_sd_pixels(edge_vals)
                    indicate[k] = float(fp)/float(sd)
                    k = 1
                    edge_vals = self.get_edge_vals(one_segment_mask_FPW,k,orig)
                    [fp, sd, fpmu, sdmu, fp_sig, sd_sig] = self.get_num_fp_sd_pixels(edge_vals)
                    indicate[k] = float(fp)/float(sd)
                    if indicate[0] > indicate[1]:
                        edge_vals = self.get_edge_vals(one_segment_mask_FPW,0,orig)
                        [fp, sd, fpmu, sdmu, fp_sig, sd_sig] = self.get_num_fp_sd_pixels(edge_vals)
                    else:
                        edge_vals = self.get_edge_vals(one_segment_mask_FPW,1,orig)
                        [fp, sd, fpmu, sdmu, fp_sig, sd_sig] = self.get_num_fp_sd_pixels(edge_vals)
                    [numsd, mfpw] = self.find_sdd_fpw(edge_vals,sd_sig)
                    tile_segs_num_sd.append(numsd)
                    tile_segs_mean_fpw.append(mfpw)
            if len(tile_segs_num_sd) == 0:
                fpw_nm[i] = 0
                total_sd[i] = 0
            else:
                fpw_nm[i] = sum(np.multiply(tile_segs_num_sd,tile_segs_mean_fpw))/sum(tile_segs_num_sd)/self.pixels_per_nm
                total_sd[i] = sum(tile_segs_num_sd)
            if len(tile_segs_area) == 0:
                gbmw_nm[i] = 0
                total_area[i] = 0
            else:
                gbmw_nm[i] = sum(np.multiply(tile_segs_area,tile_segs_gbmw))/sum(tile_segs_area)/self.pixels_per_nm
                total_area[i] = sum(tile_segs_area)

        self.save_csv(images_names, gbmw_nm, fpw_nm)

    def save_csv(self, images_names, gbmw_nm, fpw_nm):
        save_dir = self.pred_root / self.fold_dir

        target_data = {}
        animal_data = {}
        for idx, img in enumerate(images_names):
            tile_name = os.path.basename(img)
            tile_data = self.img_data.loc[self.img_data['tile_index']==tile_name]
            tile_data = tile_data.loc[tile_data['input_directory']=='test/inputs'].iloc[0]
            animal = tile_data[1].split('-')[0]
            gbmw = tile_data[5]
            fpw = tile_data[6]

            if animal not in target_data:
                new_animal = {'gbmw': [], 'fpw': []}
                new_target = {'gbmw': [], 'fpw': []}
                animal_data[animal] = new_animal
                target_data[animal] = new_target

            if gbmw != -1 and not math.isnan(gbmw):
                target_data[animal]['gbmw'].append(gbmw)
                animal_data[animal]['gbmw'].append(gbmw_nm[idx])
            if fpw != -1 and not math.isnan(fpw):
                target_data[animal]['fpw'].append(fpw)
                animal_data[animal]['fpw'].append(fpw_nm[idx])
        
        target_data_csv = pd.DataFrame(columns=['animal', 'fpw', 'gbmw'])
        animal_data_csv = pd.DataFrame(columns=['animal', 'fpw', 'gbmw'])

        for key in target_data.keys():
            target_mfpw = np.mean(target_data[key]['fpw'])
            target_mgbmw = np.mean(target_data[key]['gbmw'])
            mfpw = np.mean(animal_data[key]['fpw'])
            mgbmw = np.mean(animal_data[key]['gbmw'])
            target_row = pd.DataFrame([[key, target_mfpw, target_mgbmw]], columns = target_data_csv.columns)
            target_data_csv = pd.concat([target_data_csv, target_row], ignore_index=True)
            animal_row = pd.DataFrame([[key, mfpw, mgbmw]], columns = target_data_csv.columns)
            animal_data_csv = pd.concat([animal_data_csv, animal_row], ignore_index=True)
        
        target_path = os.path.join(save_dir, 'target_data.csv')
        pred_path = os.path.join(save_dir, 'pred_data.csv')
        target_data_csv.to_csv(target_path)
        animal_data_csv.to_csv(pred_path)
