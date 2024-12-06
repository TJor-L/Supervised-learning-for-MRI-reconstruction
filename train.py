import torch
import argparse
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import matplotlib.pyplot as plt
from model import UNet, MRIReconstructionNet, ResNet
from dataset import MRIDataset
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


def evaluate(model, dataloader, device, phase="Validation"):
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for x_i, x_hat in dataloader:
            x_hat, x_i = x_hat.to(device), x_i.to(device)
            x_pred = model(x_hat).cpu().numpy()
            x_i = x_i.cpu().numpy()

            for pred, gt in zip(x_pred, x_i):
                psnr_values.append(
                    psnr(gt, pred, data_range=gt.max() - gt.min()))
                ssim_channel_values = [
                    ssim(gt[c], pred[c], data_range=gt[c].max() - gt[c].min())
                    for c in range(gt.shape[0])
                ]
                ssim_values.append(np.mean(ssim_channel_values))

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"{phase} - Average PSNR: {avg_psnr:.2f}")
    print(f"{phase} - Average SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, save_path="best_checkpoint.pth"):
    best_val_psnr = -np.inf
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_i, x_hat in train_loader:
            x_hat, x_i = x_hat.to(device), x_i.to(device)
            optimizer.zero_grad()
            x_pred = model(x_hat)
            loss = criterion(x_pred, x_i)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        val_psnr, val_ssim = evaluate(
            model, val_loader, device, phase="Validation")
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr
            }, save_path)


def predict(model, dataloader, device, output_dir="predictions"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (x_i, x_hat) in enumerate(dataloader):
            x_hat = x_hat.to(device)
            x_pred = model(x_hat).cpu().numpy()
            x_i = x_i.numpy()
            x_hat = x_hat.cpu().numpy()

            for b in range(x_pred.shape[0]):
                pred_re = x_pred[b, 0]
                pred_im = x_pred[b, 1]
                pred_abs = np.abs(pred_re + 1j * pred_im)

                gt_re = x_i[b, 0]
                gt_im = x_i[b, 1]
                gt_abs = np.abs(gt_re + 1j * gt_im)

                x_hat_re = x_hat[b, 0]
                x_hat_im = x_hat[b, 1]
                x_hat_abs = np.abs(x_hat_re + 1j * x_hat_im)

                fig, axs = plt.subplots(1, 3, figsize=(15, 7))
                axs[0].imshow(gt_abs, cmap='gray')
                axs[0].set_title("x_i")
                axs[0].axis('off')

                axs[1].imshow(x_hat_abs, cmap='gray')
                axs[1].set_title("x_hat")
                axs[1].axis('off')

                axs[2].imshow(pred_abs, cmap='gray')
                axs[2].set_title("Prediction")
                axs[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(
                    output_dir, f"comparison_{idx}_{b}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="config.yaml", help="Path to config file.")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "predict"], help="Mode: train or predict.")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint.pth", help="Path to checkpoint.")
    parser.add_argument("--output_dir", type=str,
                        default="results", help="Directory to save predictions.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    data_file = config["data_file"]
    target_size = tuple(config["target_size"])
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    validation_split = config["validation_split"]
    test_split = config["test_split"]
    save_path = config["save_path"]
    model_name = config["model_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == "U-Net":
        model = UNet().to(device)
    elif model_name == "MRIReconstructionNet":
        model = MRIReconstructionNet().to(device)
    elif model_name == "ResNet":
        print("ResNet")
        model = ResNet().to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    dataset = MRIDataset(data_file, target_size=target_size)
    total_size = len(dataset)
    val_size = int(validation_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if args.mode == "train":

        train_model(model, train_loader, val_loader, criterion,
                    optimizer, device, num_epochs=num_epochs, save_path=args.checkpoint)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        evaluate(model, test_loader, device, phase="Test")

    elif args.mode == "predict":

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        predict(model, test_loader, device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
