import os
import util
import torch
from torchvision import transforms


from autoencoder_dataset import AutoencoderDataset

if __name__ == "__main__":

    image_size = 224
    model_path = "results/Correct-loss-2/2021-04-10_14_33_28_411982"
    train_xform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    eval_dataset = AutoencoderDataset("train", train_xform)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)

    autoencoder, _ = util.load_model(model_path)
    autoencoder.eval()
    modelName = os.path.basename(model_path)


    for i, [inputs, outputs] in enumerate(eval_dataloader, 0):
        print(f"{i}/{len(eval_dataloader)}")
        util.save_progress_image(autoencoder, inputs, i)

