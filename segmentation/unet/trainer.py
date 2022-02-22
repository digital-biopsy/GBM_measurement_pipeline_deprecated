# -*- coding: utf-8 -*-
"""trainer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FXloJXP_BImJEQT94-1TUoD4731aXLBV
"""

import numpy as np
import torch
import wandb


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_dataLoader: torch.utils.data.Dataset,
                 validation_dataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 verbose: bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_dataLoader = training_dataLoader
        self.validation_dataLoader = validation_dataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        self.verbose = verbose

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        self.tqdm = tqdm
        self.trange = trange

    def run_trainer(self):
        progressbar = self.trange(self.epochs, desc='Progress')
        for i in progressbar:
            self.epoch += 1
            # start training
            self._train()
            # start validation
            if self.validation_dataLoader is not None:
                self._validate()
            
            wandb.log({
                "validation_loss": self.val_loss,
                "training_loss": self.train_loss
                })

            # Learning rate scheduler block
            if self.lr_scheduler is not None:
                if self.validation_dataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate


    def _train(self):
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = self.tqdm(enumerate(self.training_dataLoader), 'Training', total=len(self.training_dataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

        self.train_loss = np.mean(train_losses).item()
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def _validate(self):
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = self.tqdm(enumerate(self.validation_dataLoader), 'Validation', total=len(self.validation_dataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

        self.val_loss = np.mean(valid_losses).item()
        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()