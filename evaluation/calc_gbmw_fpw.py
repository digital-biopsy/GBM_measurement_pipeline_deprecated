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


class GBMW_FPW():
    def __init__(self):
        self.FPW_blur_const = 9; #blur width
        self.GBMW_blur_const = 3; #blur width
        self.pixels_per_nm = 0.12 #convert to nm
    
    def get_filenames_of_path(self, path: pathlib.Path, ext: str = "*"):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [file for file in path.glob(ext) if file.is_file()]
        return sorted(filenames)
    
    def get_images(self, root):
        images_names_pred = self.get_filenames_of_path(root / "outputs")
        predicted = [imread(img_name) for img_name in images_names_pred]
        images_names_orig = self.get_filenames_of_path(root / "inputs")
        original = [imread(img_name) for img_name in images_names_orig]
        return predicted, original

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

    def get_num_fp_sd_pixels(edge_vals):
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
        smoothvals = savgol_filter(edge_vals,15,5) ####
        indexes = peakutils.peak.indexes(smoothvals,thres=sd_sig/smoothvals.mean(),min_dist=15) ####
        numsd = len(indexes)
        mfpw = len(edge_vals)/numsd #could subtract one from 
        return numsd, mfpw

    def find_area_gbmw(self, seg_mask):
        area = sum(sum(seg_mask))
        skeleton = skeletonize(seg_mask)
        length = sum(sum(skeleton))
        gbmw = area/length
        return area,gbmw

    def calc(self):
        #Combined main script (ignore RunTimeWarnings)
        root = pathlib.Path.cwd() / "data" / "kfold" / "fold_1"
        [predicted, original] = self.get_images(root)
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