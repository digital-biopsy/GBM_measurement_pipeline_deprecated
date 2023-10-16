import os
import cv2
import torch
import shutil
import pathlib
import numpy as np
import albumentations
import matplotlib as mpl
from termcolor import colored
import matplotlib.pyplot as plt
from re import S
from transformations import (
    AlbuSeg2d,
    normalize_01,
    re_normalize,
    ComposeDouble,
    create_dense_target,
    FunctionWrapperDouble,
)
from unet import UNet
from trainer import Trainer
from inference import predict
from skimage.io import imread
from torchsummary import summary
from visual import plot_training
from skimage.transform import resize
from matplotlib.pyplot import figure
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from customdatasets3 import SegmentationDataSet3
from sklearn.model_selection import train_test_split

class UnetSeg:
    def __init__(self, 
                 data_path,
                 epochs,
                 weight,
                 fit_steps,
                 out_channels,
                 batch_size,
                 channel_dims,
                 device,
                 criterion,
                 start_filters,
                 loss_func,
                 verbose=False,
                 current_dir=''):
        self.path = data_path
        self.verbose = verbose
        self.init = False
        self.current_dir = current_dir

        # hyperparameters
        self.epochs = epochs
        self.cross_entropy_weight = weight
        self.fit_steps = fit_steps
        self.device = device
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.criterion = criterion
        self.start_filters = start_filters
        self.loss_func = loss_func

        # constants
        self.channel_dims = channel_dims
        self.out_shape = 512
        self.learning_rate = 0.01

    
    def load_and_augment(self):
        """Load and preprocess the training images"""
        print(colored(('#'*25 + ' Loading and Augmenting Images ' + '#'*25), 'green'))
        root = pathlib.Path.cwd()/self.path

        def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
            """Returns a list of files in a directory/path. Uses pathlib."""
            filenames = [file for file in path.glob(ext) if file.is_file()]
            def get_key(fp):
                filename = os.path.splitext(os.path.basename(fp))[0]
                int_part = filename.split()[0]
                return int(int_part)
            return sorted(filenames, key=get_key)

        # for some reason the augmented data is (0,255) and won't normalize...
        # input and target files
        inputs = get_filenames_of_path(root / "inputs")
        targets = get_filenames_of_path(root / "labels")

        # pre-transformations
        pre_transforms = ComposeDouble(
            [
                FunctionWrapperDouble(
                    resize,
                    input=True,
                    target=False,
                    output_shape=(self.out_shape, self.out_shape, self.channel_dims) #3
                ),
                FunctionWrapperDouble(
                    resize,
                    input=False,
                    target=True,
                    output_shape=(self.out_shape, self.out_shape),
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ),
            ]
        )

        # training transformations and augmentations
        transforms_training = ComposeDouble(
            [
                AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
                FunctionWrapperDouble(create_dense_target, input=False, target=True),
                FunctionWrapperDouble(
                    np.moveaxis, input=True, target=False, source=-1, destination=0
                ),
                FunctionWrapperDouble(normalize_01),
            ]
        )

        # validation transformations
        transforms_validation = ComposeDouble(
            [
                FunctionWrapperDouble(
                    resize,
                    input=True,
                    target=False,
                    output_shape=(self.out_shape, self.out_shape, self.channel_dims) #3
                ),
                FunctionWrapperDouble(
                    resize,
                    input=False,
                    target=True,
                    output_shape=(self.out_shape, self.out_shape),
                    order=0,
                    anti_aliasing=False,
                    preserve_range=True,
                ),
                FunctionWrapperDouble(create_dense_target, input=False, target=True),
                FunctionWrapperDouble(
                    np.moveaxis, input=True, target=False, source=-1, destination=0
                ),
                FunctionWrapperDouble(normalize_01),
            ]
        )

        # random seed
        random_seed = 42

        # split dataset into training set and validation set
        train_size = 0.8  # 80:20 split

        inputs_train, inputs_valid = train_test_split(
            inputs, random_state=random_seed, train_size=train_size, shuffle=True
        )

        targets_train, targets_valid = train_test_split(
            targets, random_state=random_seed, train_size=train_size, shuffle=True
        )

        # inputs_train, inputs_valid = inputs[:80], inputs[80:]
        # targets_train, targets_valid = targets[:80], targets[:80]

        # dataset training
        dataset_train = SegmentationDataSet3(
            inputs=inputs_train,
            targets=targets_train,
            transform=transforms_training,
            use_cache=True,
            pre_transform=pre_transforms,
        )

        # dataset validation
        dataset_valid = SegmentationDataSet3(
            inputs=inputs_valid,
            targets=targets_valid,
            transform=transforms_validation,
            use_cache=True,
            pre_transform=pre_transforms,
        )

        # dataloader training
        self.dataloader_training = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)

        # dataloader validation
        self.dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=self.batch_size, shuffle=True)

        if self.verbose:
            x, y = next(iter(self.dataloader_training))
            print(f"x = shape: {x.shape}; type: {x.dtype}")
            print(f"x = min: {x.min()}; max: {x.max()}")
            print(f"y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}")
            #visualize some data as sanity check
            for i in range(0,8): 
                image = imread(inputs_train[i]) #inputs_train, inputs_valid, targets_train, or targets_valid
                plt.subplot(2,4,i+1)
                plt.imshow(image)
                plt.gray()
            plt.show()

    def initialize_model(self):
        device = torch.device(self.device)
        self.model = UNet(in_channels=self.channel_dims, #3
             out_channels=self.out_channels,
             n_blocks=4,
             start_filters=self.start_filters,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)


        summary(self.model, (1, 512, 512), device=self.device)

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) #learning rate
        
        # done initialization
        self.init = True

    def train_model(self):
        # Train the model and find the best learning rate.
        print(colored(('#'*25 + ' Start Training ' + '#'*25), 'green'))
        # starts training
        device = torch.device(self.device)
        trainer = Trainer(model=self.model,
                        device=device,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        training_dataLoader=self.dataloader_training,
                        validation_dataLoader=self.dataloader_validation,
                        lr_scheduler=None,
                        epochs=self.epochs,
                        epoch=0,
                        notebook=False,
                        verbose=self.verbose,
                        current_dir=self.current_dir)

        # start training
        training_losses, validation_losses, lr_rates = trainer.run_trainer()
        
        # # find best learning rate
        # self.find_lr(training_losses, validation_losses, lr_rates)

    def find_lr(self, training_losses, validation_losses, lr_rates):
        # learning rate finding script
        from lr_rate_finder import LearningRateFinder

        lrf = LearningRateFinder(self.model, self.criterion, self.optimizer, device)
        lrf.fit(self.dataloader_training, steps=self.fit_steps)
        lrf.plot()

        fig = plot_training(
            training_losses,
            validation_losses,
            lr_rates,
            gaussian=True,
            sigma=1,
            figsize=(10, 4),
        )

    
    def load_and_predict(self, model_name, out_channels):
        # Load and preprocess validation images
        print(colored(('#'*25 + ' Loading Validation Images ' + '#'*25), 'green'))
        # root directory
        root = pathlib.Path.cwd() / "data" / "test"

        def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
            """Returns a list of files in a directory/path. Uses pathlib."""
            filenames = [file for file in path.glob(ext) if file.is_file()]
            def get_key(fp):
                filename = os.path.splitext(os.path.basename(fp))[0]
                int_part = filename.split()[0]
                return int(int_part)
            return sorted(filenames, key=get_key)

        # input and target files
        images_names = get_filenames_of_path(root / "inputs")
        targets_names = get_filenames_of_path(root / "labels")

        # read images and store them in memory
        images = [imread(img_name) for img_name in images_names]
        targets = [imread(tar_name) for tar_name in targets_names]

        # Resize images and targets
        images_res = [resize(img, (self.out_shape, self.out_shape, self.channel_dims)) for img in images] #3
        resize_kwargs = {"order": 0, "anti_aliasing": False, "preserve_range": True}
        targets_res = [resize(tar, (self.out_shape, self.out_shape), **resize_kwargs) for tar in targets]

        # device and weight
        device = torch.device(self.device)
        model_path = 'models/' + model_name + '.pt'

        model_weights = torch.load(pathlib.Path.cwd() / model_path, map_location=device)
        self.model.load_state_dict(model_weights)

        # preprocess function
        def preprocess(img: np.ndarray):
            img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
            img = normalize_01(img)  # linear scaling to range [0-1]
            img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
            img = img.astype(np.float32)  # typecasting to float32
            return img

        # postprocess function
        def postprocess(img: torch.tensor):
            if out_channels > 1:
                img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
            img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
            img = np.where(img > 0.5, 1, 0)
            img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
            img = normalize_01(img)
            # img = re_normalize(img)  # scale it to the range [0-255]
            return img

        output = [predict(img, self.model, preprocess, postprocess, device) for img in images_res]
        self.save_predictions(output, targets_res, model_name)
    
    def save_predictions(self, output, targets_res, model_name):
        # Create dirs to save predictions
        abs_path = pathlib.Path.cwd() / 'pred' / model_name
        save_path = 'pred/' + model_name
        if os.path.exists(abs_path): shutil.rmtree(abs_path) # clear path
        os.makedirs(abs_path)
        
        for i in range(len(output)):
            cv2.imwrite(save_path + '/' + str(i+1) + '.jpg', output[i])