# Implementation of W-Net: A Deep Model for Fully Unsupervised Image Segmentation
# in Pytorch.
# Author: Griffin Bishop

from __future__ import division
from __future__ import print_function

import argparse
import inspect
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

import util
from autoencoder_dataset import AutoencoderDataset
from config import Config
from model import WNet
from soft_n_cut_loss import soft_n_cut_loss
from soft_n_cut_loss2 import NCutLoss2D

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
    val_xform = transforms.Compose([
        # transforms.CenterCrop(224),
        # transforms.Resize((64, 64)),
        transforms.Resize((image_size, image_size)),
        # transforms.CenterCrop(config.input_size),
        transforms.ToTensor()
    ])
    train_xform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    #TODO: Load validation segmentation maps too  (for evaluation purposes)
    train_dataset = AutoencoderDataset("train", train_xform)
    val_dataset   = AutoencoderDataset("val", val_xform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, pin_memory=True)

    data = next(iter(train_dataloader))
    mean, std = data[0].mean(), data[0].std()

    train_xform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_dataset = AutoencoderDataset("train", train_xform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    train_iter = iter(train_dataloader)

    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=4, num_workers=4, shuffle=False, pin_memory=True)
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

    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()

    lr = 0.003
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    # if config.debug:
    #     print(autoencoder)
    util.enumerate_params([autoencoder])


    # summary(autoencoder, input_size=(3, 96, 96))
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    # image_size = (224, 224)
    # data_transforms = transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor()
    # ])
    # dataset = datasets.ImageFolder("datasets/BSDS300", transform=train_xform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True, pin_memory=True)
    # data_cuda = [x[0].cuda() for x in iter(dataloader)][0]


    ###################################
    #          Loss Criterion         #
    ###################################

    def reconstruction_loss(x, x_prime):
        # binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
        # return binary_cross_entropy
        return F.mse_loss(x, x_prime, reduction='sum')


    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.train()

    progress_images, progress_expected = next(iter(val_dataloader))

    ncutloss_layer = NCutLoss2D()

    start_time = datetime.now()
    softncut_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    for iteration in range(0, config.num_epochs):

        try:
            inputs, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            inputs, _ = next(train_iter)


        inputs  = inputs.cuda()

        optimizer.zero_grad()

        # segmentations, reconstructions = autoencoder(inputs)
        segmentations = autoencoder.forward_encoder(inputs)

        l_soft_n_cut = ncutloss_layer(segmentations, inputs)
        # l_soft_n_cut = soft_n_cut_loss(inputs, segmentations)

        softncut_loss_sum += l_soft_n_cut.item()

        l_soft_n_cut.backward(retain_graph=True)
        # print(l_soft_n_cut.item())

        reconstructions = autoencoder.forward_decoder(segmentations)
        l_reconstruction = reconstruction_loss(inputs, reconstructions)

        reconstruction_loss_sum += l_reconstruction.item()

        l_reconstruction.backward()

        # loss = (l_reconstruction + l_soft_n_cut)
        # loss.backward(retain_graph=False) # We only need to do retain graph =true if we're backpropping from multiple heads
        # l_reconstruction.backward()

        optimizer.step()

        scheduler.step()

        print(f"\r{iteration}/{config.num_epochs}", end='')


        if config.showSegmentationProgress and iteration % 100 == 0: # If first batch in epoch
            print("")
            epoch_duration = datetime.now() - start_time
            print(f"Epoch duration: {epoch_duration.seconds}")
            start_time = datetime.now()

            softncut_loss_sum /= 100

            reconstruction_loss_sum /= 100
            reconstruction_loss_sum /= config.batch_size

            print(f"Soft N Cut Loss: {softncut_loss_sum:.6f}")
            print(f"Reconstruction Loss: {reconstruction_loss_sum:.6f}")


            softncut_loss_sum = 0.0
            reconstruction_loss_sum = 0.0


            util.save_progress_image(autoencoder, progress_images, iteration)
            optimizer.zero_grad() # Don't change gradient on validation

        # if epoch % 1000 == 0:
        #     lr /= 10
        #     optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

        # epoch_loss = running_loss / len(train_dataloader.dataset)
        # print("")
        # epoch_duration = datetime.now() - start_time
        # print(f"Epoch duration: {epoch_duration.seconds}")
        # print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
        # print(f"Epoch {epoch}")



    if config.saveModel:
        util.save_model(autoencoder, modelName)



if __name__ == "__main__":
    main()
