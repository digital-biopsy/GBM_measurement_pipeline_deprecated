from skimage.io import imread
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from skimage import filters
from skimage import morphology
import cv2
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
from sklearn.metrics import jaccard_score
import shutil
from PIL import Image

class GBMW_FPW():
    def __init__(self):
        self.FPW_blur_const = 9; #blur width
        self.GBMW_blur_const = 3; #blur width
        self.pixels_per_nm = 0.12 #convert to nm
        self.img_dir = pathlib.Path.cwd() / "data" / "test" / "inputs"
        self.label_dir = pathlib.Path.cwd() / "data" / "test" / "labels"
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
        # pred_dir = pathlib.Path.cwd() / "data" / "test" / "labels"
        images_names = self.get_filenames_of_path(self.img_dir)
        labels_names = self.get_filenames_of_path(self.label_dir)
        predicted_names = self.get_filenames_of_path(pred_dir)
        # read images and store them in memory
        images = [imread(img_name) for img_name in images_names]
        predicted = [np.invert(imread(tar_name)) for tar_name in predicted_names]
        labels = [imread(label_name) for label_name in labels_names]
        return images_names, predicted, images, labels

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
        edges = cv2.Canny(onesegmentmask.astype(np.uint8),0,0.15)
        label_img = label(edges, connectivity=edges.ndim)
        props = regionprops(label_img)
        num_edges = len(props)
        return num_edges

    def get_edge_vals(self, onesegmentmask,k,orig):
        edges = cv2.Canny(onesegmentmask.astype(np.uint8),0,0.15)
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

    def calc_manager(self, cur_fold, cur_model):
        self.GBMW_blur_const = 4; #blur width
        self.calc(cur_fold, cur_model)

    def save_results(self, cur_fold, cur_model):
        self.fold_dir = cur_fold
        self.model_dir = cur_model
        [images_names, predicted, original, labels] = self.get_images()
        save_dir = self.pred_root / self.fold_dir / 'contrasts'
        blur_dir = save_dir / 'blurred'
        orig_dir = save_dir / 'original'
        pred_dir = save_dir / 'predicted'
        label_dir = save_dir / 'label'

        self.reset_dir([save_dir, blur_dir, orig_dir, pred_dir, label_dir])

        img_num = 20
        if len(predicted) >= img_num*2:
            # fig, axs = plt.subplots(20, 4, figsize=(20,60))
            for i in range(img_num):
                predict = predicted[i*2]
                label = labels[i*2]
                orig = original[i*2]
                flip_predict = np.invert(predict)

                masked_FPW = self.blur_mask(flip_predict, self.FPW_blur_const)
                masked_GBMW = self.blur_mask(flip_predict, self.GBMW_blur_const)
                invlabel = 1 - (label//255)
                pred_bin = np.rint(flip_predict//255).astype(int)
                # print('Image')
                # print('label', np.unique(invlabel), np.mean(invlabel))
                # print('pred_bin', np.unique(pred_bin), np.mean(pred_bin))
                # print('masked_GBMW', np.unique(masked_GBMW), np.mean(masked_GBMW))

                pred_contrast = self.generate_jaccard_image(pred_bin, invlabel)
                blur_contrast = self.generate_jaccard_image(masked_GBMW, invlabel)
                image_name = f'{i}.png'
                
                cv2.imwrite(str(pred_dir/image_name), pred_contrast)
                cv2.imwrite(str(blur_dir/image_name), blur_contrast)
                cv2.imwrite(str(orig_dir/image_name), orig)
                cv2.imwrite(str(label_dir/image_name), invlabel*255)
                # contrast = np.invert(label - np.invert(masked_GBMW))//2

                # r = i//2
                # c = i%2
                # axs[r,2*c].imshow(contrast)
                # # axs[r,2*c].imshow(contrast, cmap='gray', vmin=0, vmax=255)
                # axs[r,2*c].axis('off')
                # # axs[r,2*c].set_title('Prediction')
                # axs[r,2*c+1].imshow(orig, cmap='gray', vmin=0, vmax=255)
                # axs[r,2*c+1].axis('off')
                # # axs[r,2*c+1].set_title('Original')

            # fig.tight_layout()
            # fig.savefig(pathlib.Path.cwd() / 'pred' / 'pred_results.png')
            # plt.close(fig)
        else:
            print("length of predictions aren't sufficient")

    def reset_dir(self, paths):
        for p in paths:
            if os.path.exists(p): shutil.rmtree(p) # clear path
            os.makedirs(p)

    def generate_jaccard_image(self, inputs, targets):
        inputs = 1 - inputs
        targets = 1 - targets

        # TP = np.bitwise_and(inputs, targets)*255
        # FP = np.bitwise_and(inputs, inv_targets).sum()
        # FN = np.bitwise_and(inv_inputs, targets).sum()
        g = inputs*255
        b = targets*255
        r = np.ones_like(targets)*255
        a = np.ones_like(targets)*255
        img = np.stack([r,g,b,a], axis=2)
        return img

    def IoU(self, inputs, targets):
        inv_inputs = 1 - inputs
        inv_targets = 1 - targets
        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        TP = np.bitwise_and(inputs, targets).sum()
        FP = np.bitwise_and(inputs, inv_targets).sum()
        FN = np.bitwise_and(inv_inputs, targets).sum()
        
        IoU = (TP)/(TP + FP + FN)
                
        return IoU

    def calc(self, cur_fold, cur_model):
        print(colored(('#'*25 + ' Calculating FPW and GBMW ' + '#'*25), 'green'))
        # Combined main script (ignore RunTimeWarnings)
        self.fold_dir = cur_fold
        self.model_dir = cur_model
        [images_names, predicted, original, labels] = self.get_images()
        mask_FPW = np.empty((len(predicted),512,512), dtype='int_')
        mask_GBMW = np.empty((len(predicted),512,512), dtype='int_')
        gbmw_nm = np.empty((len(predicted)), dtype='object')
        total_area = np.empty((len(predicted)), dtype='object')
        fpw_nm = np.empty((len(predicted)), dtype='object')
        total_sd = np.empty((len(predicted)), dtype='object')
        jaccard = np.empty((len(predicted)), dtype='object')
        jaccard_blur = np.empty((len(predicted)), dtype='object')
        assert len(predicted) == len(labels), 'labels length and prediction length mismatch'
        for i in range(0,len(predicted)):
            predict = predicted[i]
            flip_predict = np.invert(predict)
            orig = original[i]
            
            masked_MFPW = self.blur_mask(flip_predict, self.FPW_blur_const)
            masked_GBMW = self.blur_mask(flip_predict, self.GBMW_blur_const)
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
                # gbmw_nm[i] = sum(tile_segs_gbmw)/len(tile_segs_gbmw)

            invlabel = 1 - (labels[i]//255)
            pred_bin = 1 - (predicted[i]//255)
            masked_GBMW_bin = mask_GBMW[i]
            # print('Pred:', np.unique(pred_bin), np.mean(pred_bin))
            # print('Label:', np.unique(invlabel), np.mean(invlabel))
            # print('GBMW_bin:', np.unique(masked_GBMW_bin), np.mean(masked_GBMW_bin))
            iou = self.IoU(invlabel, pred_bin)
            j_idx = jaccard_score(invlabel, pred_bin, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
            j_idx_blur = jaccard_score(invlabel, masked_GBMW_bin, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
            # print('Jaccard:', j_idx, 'IoU:', iou)
            jaccard[i] = j_idx
            jaccard_blur[i] = j_idx_blur

        self.save_csv(images_names, gbmw_nm, fpw_nm, jaccard, jaccard_blur)

    def save_csv(self, images_names, gbmw_nm, fpw_nm, jaccard, jaccard_blur):
        save_dir = self.pred_root / self.fold_dir

        target_animal_data = {}
        predict_animal_data = {}
        target_image_data = {}
        predict_image_data = {}
        for idx, img in enumerate(images_names):
            tile_name = os.path.basename(img)
            tile_data = self.img_data.loc[self.img_data['tile_index']==tile_name]
            tile_data = tile_data.loc[tile_data['input_directory']=='test/inputs'].iloc[0]
            animal = tile_data[1].split('-')[0]
            image = tile_data[1].split('.jpg')[0]
            gbmw = tile_data[5]
            fpw = tile_data[6]
            j_idx = jaccard[idx]
            j_idx_blur = jaccard_blur[idx]

            if animal not in target_animal_data:
                new_animal_target = {'gbmw': [], 'fpw': []}
                new_animal_pred = {'gbmw': [], 'fpw': [], 'jaccard': [], 'jaccard_blur': []}
                target_animal_data[animal] = new_animal_target
                predict_animal_data[animal] = new_animal_pred

            if image not in target_image_data:
                new_image_target = {'gbmw': [], 'fpw': []}
                new_image_predict = {'gbmw': [], 'fpw': [], 'jaccard': [], 'jaccard_blur': []}
                target_image_data[image] = new_image_target
                predict_image_data[image] = new_image_predict

            # print('img', tile_data[1], 'gbmw', gbmw_nm[idx], 'gbmw target', gbmw)
            # print('img', tile_data[1], 'fpw', fpw_nm[idx], 'fpw target', fpw)
            predict_image_data[image]['jaccard'].append(j_idx)
            predict_animal_data[animal]['jaccard'].append(j_idx)

            predict_image_data[image]['jaccard_blur'].append(j_idx_blur)
            predict_animal_data[animal]['jaccard_blur'].append(j_idx_blur)

            if gbmw != -1 and not math.isnan(gbmw) and gbmw_nm[idx] != 0 and not math.isnan(gbmw_nm[idx]):
                target_animal_data[animal]['gbmw'].append(gbmw)
                predict_animal_data[animal]['gbmw'].append(gbmw_nm[idx])
                target_image_data[image]['gbmw'].append(gbmw)
                predict_image_data[image]['gbmw'].append(gbmw_nm[idx])
            
            # print(f'Animal: {animal}, Image: {image}, fpw: {fpw}, pred: {fpw_nm[idx]}')
            if fpw != -1 and not math.isnan(fpw) and fpw_nm[idx] != 0 and not math.isnan(fpw_nm[idx]):
                target_animal_data[animal]['fpw'].append(fpw)
                predict_animal_data[animal]['fpw'].append(fpw_nm[idx])
                target_image_data[image]['fpw'].append(fpw)
                predict_image_data[image]['fpw'].append(fpw_nm[idx])
        
        animal_data_csv = pd.DataFrame(columns=['animal', 'fpw', 'target fpw', 'fpw percent error', 'gbmw', 'target gbmw', 'gbmw percent error', 'mean jaccard index', 'mean jaccard index blurred'])
        image_data_csv = pd.DataFrame(columns=['image', 'fpw', 'target fpw', 'fpw percent error', 'gbmw', 'target gbmw', 'gbmw percent error', 'mean jaccard index', 'mean jaccard index blurred'])

        for key in target_animal_data.keys():
            target_mfpw = np.mean(target_animal_data[key]['fpw'])
            target_mgbmw = np.mean(target_animal_data[key]['gbmw'])
            mfpw = np.mean(predict_animal_data[key]['fpw'])
            # print(f'Anmimal {key} target fpw:', target_animal_data[key]['fpw'], target_mfpw)
            # print(f'Anmimal {key} predet fpw:', predict_animal_data[key]['fpw'], mfpw)
            mgbmw = np.mean(predict_animal_data[key]['gbmw'])
            mjaccard = np.mean(predict_animal_data[key]['jaccard'])
            mjaccardblur = np.mean(predict_animal_data[key]['jaccard_blur'])
            animal_row = pd.DataFrame([[key, mfpw, target_mfpw, 1 - mfpw/target_mfpw, mgbmw, target_mgbmw, 1 - mgbmw/target_mgbmw, mjaccard, mjaccardblur]], columns = animal_data_csv.columns)
            animal_data_csv = pd.concat([animal_data_csv, animal_row], ignore_index=True)

        for key in target_image_data.keys():
            target_mfpw = np.mean(target_image_data[key]['fpw'])
            target_mgbmw = np.mean(target_image_data[key]['gbmw'])
            mfpw = np.mean(predict_image_data[key]['fpw'])
            # print(f'Image {key} target fpw:', target_image_data[key]['fpw'], target_mfpw)
            # print(f'Image {key} predet fpw:', predict_image_data[key]['fpw'], mfpw)
            mgbmw = np.mean(predict_image_data[key]['gbmw'])
            mjaccard = np.mean(predict_image_data[key]['jaccard'])
            mjaccardblur = np.mean(predict_image_data[key]['jaccard_blur'])
            image_row = pd.DataFrame([[key, mfpw, target_mfpw, 1 - mfpw/target_mfpw, mgbmw, target_mgbmw, 1 - mgbmw/target_mgbmw, mjaccard, mjaccardblur]], columns = image_data_csv.columns)
            image_data_csv = pd.concat([image_data_csv, image_row], ignore_index=True)
        
        animal_path = os.path.join(save_dir, 'animal_data_%.1f.csv'%self.GBMW_blur_const)
        image_path = os.path.join(save_dir, 'image_data_%.1f.csv'%self.GBMW_blur_const)
        animal_data_csv.to_csv(animal_path)
        image_data_csv.to_csv(image_path)
