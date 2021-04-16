import os
import util
import torch
from torchvision import transforms


import numpy as np
from mss import mss
from PIL import Image
import cv2

from config import Config
from autoencoder_dataset import AutoencoderDataset

if __name__ == "__main__":

    # image_size = 224
    # model_path = "results/Correct-loss-2/2021-04-10_14_33_28_411982"
    # train_xform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),
    #     transforms.ToTensor()
    # ])
    # eval_dataset = AutoencoderDataset("train", train_xform)
    # eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    #
    # autoencoder, _ = util.load_model(model_path)
    # autoencoder.eval()
    # modelName = os.path.basename(model_path)
    #
    #
    # for i, [inputs, outputs] in enumerate(eval_dataloader, 0):
    #     print(f"{i}/{len(eval_dataloader)}")
    #     util.save_progress_image(autoencoder, inputs, i)

    mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}

    sct = mss()

    model_path = "results/Correct-loss-2/2021-04-10_14_33_28_411982"
    autoencoder, _ = util.load_model(model_path)
    autoencoder.eval()
    torch.no_grad()

    config = Config()

    while 1:

        img = np.array(sct.grab(sct.monitors[1]))
        img = cv2.resize(img, (224, 224))

        cuda_img = torch.from_numpy(img).cuda()
        cuda_img = cuda_img.permute(2, 0, 1)[:3, :, :].unsqueeze(0) / 255
        segmentation = autoencoder.forward_encoder(cuda_img)
        pixels = torch.argmax(segmentation[0], axis=0).float() / config.k # to [0,1]
        resized_pixels = cv2.resize(pixels.detach().cpu().numpy(), (1280, 720))
        colored_pixels = cv2.applyColorMap(np.uint8(resized_pixels * 255), cv2.COLORMAP_SUMMER)
        cv2.imshow('test', colored_pixels)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
