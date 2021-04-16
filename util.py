import os, shutil
from config import Config
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

config = Config()

import torch.nn as nn
from crfseg import CRF
# crf = nn.Sequential(
#     CRF(n_spatial_dims=1)
# )
# crf.cuda()

# Clear progress images directory
def clear_progress_dir(): # Or make the dir if it does not exist
    if not os.path.isdir(config.segmentationProgressDir):
        os.mkdir(config.segmentationProgressDir)
    else: # Clear the directory
        for filename in os.listdir(config.segmentationProgressDir):
            filepath = os.path.join(config.segmentationProgressDir, filename)
            os.remove(filepath)

def enumerate_params(models):
	num_params = 0
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				num_params += param.numel()
	print(f"Total trainable model parameters: {num_params}")

def save_model(autoencoder, modelName):
    path = os.path.join("./models/", modelName.replace(":", " ").replace(".", " ").replace(" ", "_"))
    torch.save(autoencoder, path)
    with open(path+".config", "a+") as f:
        f.write(str(config))
        f.close()

def load_model(model):
    with open(model+".config", "a+") as f:
        config = f.read()

    return (torch.load(model), config)

def save_progress_image(autoencoder, progress_images, epoch):
    if not torch.cuda.is_available():
        segmentation, reconstructions = autoencoder(progress_images)
    else:
        segmentation, reconstructions = autoencoder(progress_images.cuda())

    # sigmoid = torch.nn.Sigmoid()

    # segmentations = [segmentations] if isinstance(segmentations, torch.Tensor) else segmentations

    f, axes = plt.subplots(3, 1, figsize=(64,64))
    # for i in range(config.val_batch_size):
    # segmentation = segmentations[i]

    pixels = torch.argmax(segmentation[0], axis=0).float() / config.k # to [0,1]
    # pixels = (pixels * 255).type(torch.IntTensor)

    axes[0].imshow(progress_images[0].permute(1, 2, 0))
    axes[1].imshow(pixels.detach().cpu())
    axes[2].imshow(reconstructions[0].detach().cpu().permute(1, 2, 0))

        # if config.variationalTranslation:
        #     axes[3, i].imshow(progress_expected[i].detach().cpu().permute(1, 2, 0))

    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"))
    # plt.show()
    plt.close(f)

def save_progress_images(autoencoder, progress_images, epoch):
    if not torch.cuda.is_available():
        segmentations, reconstructions = autoencoder(progress_images)
    else:
        segmentations, reconstructions = autoencoder(progress_images.cuda())

    with torch.no_grad():
        autoencoder.eval()
        segmentations_eval, reconstructions_eval = autoencoder(progress_images.cuda())
        autoencoder.train()

    f, axes = plt.subplots(4, config.val_batch_size, figsize=(64,64))
    for i in range(config.val_batch_size):
        segmentation = segmentations[i]
        segmentation_eval = segmentations_eval[i]

        pixels = torch.argmax(segmentation, axis=0).float() / config.k # to [0,1]
        eval_pixels = torch.argmax(segmentation_eval, axis=0).float() / config.k # to [0,1]

        axes[0, i].imshow(progress_images[i].permute(1, 2, 0))
        axes[1, i].imshow(pixels.detach().cpu())
        axes[2, i].imshow(eval_pixels.detach().cpu())
        axes[3, i].imshow(reconstructions[i].detach().cpu().permute(1, 2, 0))

    plt.savefig(os.path.join(config.segmentationProgressDir, str(epoch)+".png"))
    plt.close(f)
