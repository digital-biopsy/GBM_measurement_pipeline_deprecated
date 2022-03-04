import torch
import wandb
import pathlib
import numpy as np
import albumentations
import matplotlib as mpl
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

class Segmentation:
    def __init__(self, data_path, epochs, weight, fit_steps, device, verbose=False):
        self.path = data_path
        self.verbose = verbose

        # hyperparameters
        self.epochs = epochs
        self.cross_entropy_weight = weight
        self.fit_steps = fit_steps
        self.device = device

        # constants
        self.channel_dims = 1
        self.out_shape = 512
        self.learning_rate = 0.01

        # initialize weights and bias
        wandb.init(
            project="digital-biopsy",
            entity="zhaoze",
            config = {
            "learning_rate": self.learning_rate,
            "cross_entropy_weight": self.cross_entropy_weight,
            "epochs": self.epochs,
            "fit_steps": self.fit_steps
        })
    
    def load_and_augment(self):
        """Load and preprocess the training images"""
        print('#'*25 + ' Loading and Augmenting Images ' + '#'*25)
        root = pathlib.Path.cwd()/self.path

        def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
            """Returns a list of files in a directory/path. Uses pathlib."""
            filenames = [file for file in path.glob(ext) if file.is_file()]
            return sorted(filenames)

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
        self.dataloader_training = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

        # dataloader validation
        self.dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=1, shuffle=True)

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
        # model
        if self.verbose:
            model = UNet(in_channels=self.channel_dims, #3
                        out_channels=2,
                        n_blocks=4,
                        start_filters=32,
                        activation='relu',
                        normalization='batch',
                        conv_mode='same',
                        dim=2)

            x = torch.randn(size=(1, 1, 512, 512), dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
            print(f"Out: {out.shape}")
            summary(model, (1, 512, 512), device="cpu")

        device = torch.device(self.device)
        self.model = UNet(in_channels=self.channel_dims, #3
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2).to(device)

        # criterion
        weights = [1, self.cross_entropy_weight]
        self.class_weights = torch.FloatTensor(weights).cuda()
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights) #CrossEntropyLoss

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate) #learning rate

    def train_model(self):
        """Train the model and find the best learning rate."""
        print('#'*25 + ' Start Training ' + '#'*25)
        device = torch.device(self.device)
        trainer = Trainer(model=self.model,
                        device=device,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        training_DataLoader=self.dataloader_training,
                        validation_DataLoader=self.dataloader_validation,
                        lr_scheduler=None,
                        epochs=self.epochs,
                        epoch=0,
                        notebook=False,
                        verbose=self.verbose)

        # start training
        training_losses, validation_losses, lr_rates = trainer.run_trainer()
        # save the model
        model_name = "unet.pt"
        torch.save(self.model.state_dict(), pathlib.Path.cwd() / model_name)

        #find best learning rate
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
    
    def load_val_images(self):
        """Load and preprocess validation images"""
        print('#'*25 + 'Loading Validation Images' + '#'*25)
        # root directory
        root = pathlib.Path.cwd() / "GBM_data_shuffled" / "test"

        def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
            """Returns a list of files in a directory/path. Uses pathlib."""
            filenames = [file for file in path.glob(ext) if file.is_file()]
            return sorted(filenames)

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

        # device
        device = torch.device(self.device)

        model_name = "unet.pt"
        model_weights = torch.load(pathlib.Path.cwd() / model_name, map_location=device)
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
            img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
            img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
            img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
            img = re_normalize(img)  # scale it to the range [0-255]
            return img

        output = [predict(img, self.model, preprocess, postprocess, device) for img in images_res]
        self.view_predictions(output, images_res, targets_res)
    
    def view_predictions(self, output, images_res, targets_res):
        """View test original images and predictions side-by-side."""
        r = 10
        c = 4
        figure(figsize=(20, 100))
        plt.gray()
        print("Output Shape: ", output[0].shape)
        print(output[0].mean())
        for i in range(0,r*c):
            predicted = output[i]
            plt.subplot(r*2,c,i*2+1)
            plt.imshow(predicted)
            plt.subplot(r*2,c,i*2+2)
            if self.channel_dims == 3:
                plt.imshow(images_res[i])
            else:
                plt.imshow(images_res[i][:,:,0])

        inv_targets_res = targets_res
        inv_output = output
        figure(figsize=(20, 30))
        for i in range(0,len(output)):
            inv_targets_res[i] = abs(normalize_01(targets_res[i]) - 1)
            inv_output[i] = abs(normalize_01(output[i]) - 1)
        for i in range(0,32): #can change which display, just also change subplot line
            plt.subplot(8,4,i+1)
            plt.imshow(inv_targets_res[i] - inv_output[i]) 
        #calculate new Jaccards index (should be ~0.7-0.8)
        jaccard = []
        for i in range(0,10): #change which images for
            jaccard.append(jaccard_score(inv_targets_res[i], inv_output[i], labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn'))
        print(f'The mean Jaccard index for test images is', np.mean(jaccard))