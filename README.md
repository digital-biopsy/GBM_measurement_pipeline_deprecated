# Digital Biopsy - Deep Segmentation
This is the segmentation tool for digital biopsy project. It also contains some useful git command instructions for the reference. The code for unet segmentation is adapted from [here](https://github.com/johschmidt42/PyTorch-2D-3D-UNet-Tutorial).

## Raw Dataset
### Dataset Structure
The data structure of the raw dataset is approx. as follows: </br>
|-folderpath/ </br>
|---stats.csv </br>
|---inputs/ </br>
|-----img-name-1.jpg </br>
|-----img-name-2.jpg </br>
|---labels/ </br>
|-----img-name-1-GBMlabels.png </br>
|-----img-name-2-GBMlabels.png.jpg </br>

### Image name and explanations
- The raw images are stored in the folder inputs, while corresponding labels where stored under the labels folder.
- The label name must be exact match of the input name and append a "-GBMlabels" to the end.
- The root path of the raw dataset is stored in the dict DATASET, which can be changed in project_root/params_meta.py.

## arg '-v
Verbosely print currently training process, helpful for debugging.
## arg '-prep'
This argument will pre-process the image. It will perform the image cropping, image shuffling, and subject-wise k-fold cross validation (CV). As the data stored in the 'stats.csv' is performed by multiple operators and lacks format consistency, it will also performed the corner case handling to exclude the data that does not contain a desired stats entry (GBMW or FPW etc.)
### Related params (params_meta.py)
kfold = 5 (0 if all in inputs, 1 if all in tests, 5 by default (80% train 20% test))
sliding_step = 300 (the step size of the sliding window after downsampling. i.e the raw img are first downsampled before performing sliding window).
### Generated Dataset
The '-prep' command will first initialize a Preprocess instance by loading all the pre-processing params. Then it reads all the raw image-label pairs in and store their absolute path in 'project_root/data/kfold/inputs-<subject-groupname>.txt'. The subject-wise kfold will requires kfold shuffle before applying cropping. Hence, Preprocessor will first shuffle the subjects and store them in 'project_root/data/kfold/fold_<n>/', append corresponding raw-image and raw-label input train/test list according to the current fold, and perform the cropping. </br>
The cropped training image tiles are stored into the 'project_root/data/labels/' and 'project_root/data/inputs/' folder. The testing labels and inputs are stored in the 'inputs' and 'labels' folder under the 'project_root/data/test/'. The validation folder is a lengacy folder, we currently use in-line validation that is randomly selected during training so this folder is no longer needed.
### Generated Dataset Structure
|-project_root/data/ </br>
|---tile_stats.csv </br>
|---kfold/ </br>
|-----fold_0/ </br>
|-------inputs.txt (all abs path of input imgs of this fold) </br>
|-------labels.txt (all abs path of labels of this fold) </br>
|-----fold_1/ </br>
|-------inputs.txt (all abs path of input imgs of this fold) </br>
|-------labels.txt (all abs path of labels of this fold) </br>
|-----inputs-<subject-groupname-1>.txt (abs paths of imgs in this group) </br>
|-----labels-<subject-groupname-1>.txt (abs paths of labels in this group) </br>
|-----inputs-<subject-groupname-2>.txt (abs paths of imgs in this group) </br>
|-----labels-<subject-groupname-2>.txt (abs paths of labels in this group) </br>
|---inputs/ (all cropped trainset inputs) </br>
|---labels/ (all cropped trainset labels) </br>
|---test/
|-----inputs/ (all cropped testset inputs) </br>
|-----labels/ (all cropped testset labels) </br>
## arg '-kfold'
*** NOTE: this will deleted the pre-trained weight matrix ***
*** NOTE: do not use any command that calls update_image_list after running this command, it will re-shuffle the kfold and cause deprecated measurement result (could be using the training fold for testing) ***
This argument will automatically do the kfold data preprocessing and the training.
### arg '-train'
*** NOTE: do not use any command that calls update_image_list after running this command, it will re-shuffle the train and cause deprecated measurement result (could be using the training fold for testing) ***
This argument train the shuffled inputs/labels pairs. It need to be used after applying '-prep'.
### arg '-pred'
This command will predict (segment) the test set using the trained_weight matrices of the corresponding fold.
### arg '-save'
This argument will visualize the predicted results w.r.t. the original image and highlight the False Positive labels and False Negative labels.
### arg '-eval'
This argument perform the GBMW and FPW measurement. It will also do the same procedure as '-save'

## To Do List
- [x] Randomize the train test split.
- [x] Add data (input/target/test) into .gitignore.
- [ ] Test if current model work on cpu devices.
- [x] Train test split already included in the code, delete the validation folder.
- [x] Intergrate the image tiling with the training script.
- [ ] Refactor code to enable training on server/local pc.
- [ ] Check the optimizer logic
- [ ] Add grayscale augmentation methods to prevent overfitting on color.
- [x] Move parameters to 'params_meta.py'
## To Run The Script
#### Commands
Data pre-processing is required before training. The data path is defined in 'params-meta.py'
```
python main.py -prep
```
Hyperparameters are defined in 'params-meta.py'
```
python main.py -train
```
The following command will predict using the trained model and save the predictions under the 'pred' folder<br>
Note: the model must align with the weight matrices that used to predict. Changing the hyperparameters might cause runtime failures.
```
python main.py -pred
```
#### Optional command(s)
The following optional command works with '-pred' and '-train'. This will verbosely print out some of the training progresses that help debugging.
```
-v
```