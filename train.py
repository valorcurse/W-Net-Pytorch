# Implementation of W-Net: A Deep Model for Fully Unsupervised Image Segmentation
# in Pytorch.
# Author: Griffin Bishop

from __future__ import division
from __future__ import print_function

import argparse
import inspect
import os
from datetime import datetime

from statistics import median

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

import util
from autoencoder_dataset import AutoencoderDataset
from config import Config
from model import WNet
from soft_n_cut_loss import soft_n_cut_loss, NCutLoss2D

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--load', metavar='of', default=None, type=str, help='model')

def main():
    args = parser.parse_args()

    print("PyTorch Version: ",torch.__version__)
    if torch.cuda.is_available():
        print("Cuda is available. Using GPU")

    config = Config()


    ###################################
    # Image loading and preprocessing #
    ###################################

    #TODO: Maybe we should crop a large square, then resize that down to our patch size?
    # For now, data augmentation must not introduce any missing pixels TODO: Add data augmentation noise
    image_size = config.input_size
    train_xform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    val_xform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    #TODO: Load validation segmentation maps too  (for evaluation purposes)
    train_dataset = AutoencoderDataset("train", train_xform)
    val_dataset   = AutoencoderDataset("val", val_xform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4, num_workers=4, shuffle=False, pin_memory=True)

    if not args.load:
        util.clear_progress_dir()

    ###################################
    #          Model Setup            #
    ###################################
    if args.load:
        autoencoder, _ = util.load_model(args.load)
        modelName = os.path.basename(args.load)

    else:
        autoencoder = WNet()
        # Use the current time to save the model at end of each epoch
        modelName = str(datetime.now())
        # autoencoder.

    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()

    iteration = 0
    lr=0.003
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(autoencoder.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

    util.enumerate_params([autoencoder])

    ###################################
    #          Loss Criterion         #
    ###################################

    def reconstruction_loss(x, x_prime):
        # binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
        return nn.MSELoss(reduction='sum')(x, x_prime)
        # return F.mse_loss(x_prime, x, reduction='mean')
        # return binary_cross_entropy


    ###################################
    #          Training Loop          #
    ###################################

    # autoencoder.train()
    autoencoder.eval()

    progress_images, progress_expected = next(iter(val_dataloader))

    ncutloss_layer = NCutLoss2D()

    start_time = datetime.now()
    softncut_loss_sum = 0.0
    reconstruction_loss_sum = 0.0

    r_losses = []
    soft_losses = []

    for epoch in range(config.num_epochs):

        for i, [inputs, outputs] in enumerate(train_dataloader, 0):

            inputs  = inputs.cuda()

            optimizer.zero_grad()

            l_soft_n_cut = ncutloss_layer(autoencoder.forward_encoder(inputs), inputs)
            l_soft_n_cut.backward(retain_graph=False)

            soft_losses.append(l_soft_n_cut.item())

            # reconstructions = autoencoder.forward_decoder(autoencoder.forward_encoder(inputs))
            reconstructions = autoencoder.forward_decoder(autoencoder.forward_encoder(inputs))
            l_reconstruction = reconstruction_loss(inputs, reconstructions)
            l_reconstruction.backward(retain_graph=False)

            r_losses.append(l_reconstruction.item())

            optimizer.step()

            scheduler.step()

            print(f"\r{iteration}/{config.num_epochs}", end='')

            if config.showSegmentationProgress and i % 100 == 0: # If first batch in epoch
                print("")
                epoch_duration = datetime.now() - start_time
                print(f"Epoch duration: {epoch_duration.seconds}")
                start_time = datetime.now()

                s_loss = median(soft_losses)
                r_loss = median(r_losses)/config.batch_size

                print(f"Soft N Cut Loss: {s_loss:.6f}")
                print(f"Reconstruction Loss: {r_loss:.6f}")
                try:
                    print(f"Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")
                except:
                    pass


                util.save_progress_images(autoencoder, progress_images, iteration)
                optimizer.zero_grad() # Don't change gradient on validation


            iteration += 1

            if config.saveModel:
                util.save_model(autoencoder, modelName)



if __name__ == "__main__":
    main()
