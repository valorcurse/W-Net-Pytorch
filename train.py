# Implementation of W-Net: A Deep Model for Fully Unsupervised Image Segmentation
# in Pytorch.
# Author: Griffin Bishop

from __future__ import division
from __future__ import print_function

import argparse
import inspect
import os
from datetime import datetime

import torch
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
    train_xform = transforms.Compose([
        # transforms.RandomCrop(224),
        transforms.Resize((64, 64)),
        # transforms.RandomCrop(config.input_size+config.variationalTranslation), # For now, cropping down to 224
        # transforms.RandomHorizontalFlip(), # TODO: Add colorjitter, random erasing
        transforms.ToTensor()
    ])
    val_xform = transforms.Compose([
        # transforms.CenterCrop(224),
        # transforms.Resize((64, 64)),
        transforms.Resize((64, 64)),
        # transforms.CenterCrop(config.input_size),
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

    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.003)
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
        binary_cross_entropy = F.binary_cross_entropy(x_prime, x, reduction='sum')
        return binary_cross_entropy


    ###################################
    #          Training Loop          #
    ###################################

    autoencoder.train()

    progress_images, progress_expected = next(iter(val_dataloader))

    ncutloss_layer = NCutLoss2D()

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        start_time = datetime.now()
        for i, [inputs, outputs] in enumerate(train_dataloader, 0):
        # for i, inputs in enumerate(data_cuda):

            frame = inspect.currentframe()          # define a frame to track
            # gpu_tracker = MemTracker(frame)         # define a GPU tracker

            print(f"\r{i}/{len(train_dataloader)}", end='')

            inputs  = inputs.cuda()

            optimizer.zero_grad()

            # segmentations, reconstructions = autoencoder(inputs)
            segmentations = autoencoder.forward_encoder(inputs)
            l_soft_n_cut = ncutloss_layer(segmentations, inputs)

            # l_soft_n_cut = soft_n_cut_loss(inputs, segmentations)
            # l_soft_n_cut = checkpoint(soft_n_cut_loss, inputs, segmentations)
            l_soft_n_cut.backward(retain_graph=True)

            reconstructions = autoencoder.forward_decoder(segmentations)
            # l_reconstruction = reconstruction_loss(
            #     inputs if config.variationalTranslation == 0 else outputs,
            #     reconstructions
            # )
            # gpu_tracker.track()
            l_reconstruction = reconstruction_loss(inputs, reconstructions)
            # gpu_tracker.track()
            l_reconstruction.backward()
            # loss = (l_reconstruction + l_soft_n_cut)
            # loss.backward(retain_graph=False) # We only need to do retain graph =true if we're backpropping from multiple heads
            # l_reconstruction.backward()
            # gpu_tracker.track()
            optimizer.step()
            # gpu_tracker.track()

            # print(modelsize(autoencoder, inputs))
            # del loss
            # torch.cuda.empty_cache()

            # if config.debug and (i%50) == 0:
            #     print(i)

            # print statistics
            # running_loss += l_reconstruction.item()


            if config.showSegmentationProgress and i == 0: # If first batch in epoch
                util.save_progress_image(autoencoder, progress_images, epoch)
                optimizer.zero_grad() # Don't change gradient on validation

        # epoch_loss = running_loss / len(train_dataloader.dataset)
        print("")
        epoch_duration = datetime.now() - start_time
        print(f"Epoch duration: {epoch_duration.seconds}")
        # print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
        print(f"Epoch {epoch}")



    if config.saveModel:
        util.save_model(autoencoder, modelName)



if __name__ == "__main__":
    main()
