import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MRIDataset(Dataset):
    def __init__(self, file_path, target_size=(256, 256)):
        with h5py.File(file_path, 'r') as f:
            self.ground_truth = f['trnOrg'][:]
            self.masks = f['trnMask'][:]
        self.target_size = target_size

    def __len__(self):
        return len(self.ground_truth)

    def pad_to_target(self, data, target_size):
        h, w = data.shape
        pad_h = (target_size[0] - h) // 2
        pad_w = (target_size[1] - w) // 2
        padding = [(pad_h, target_size[0] - h - pad_h),
                   (pad_w, target_size[1] - w - pad_w)]
        padded_data = np.pad(data, padding, mode='constant', constant_values=0)
        return padded_data

    def get_x_hat(self, x, P_i):
        F_x = np.fft.fft2(x)
        y_i = P_i * F_x
        x_hat = np.fft.ifft2(y_i)
        return x_hat

    def to_2channel_tensor(self, data):
        real = np.real(data)
        imag = np.imag(data)
        stacked = np.stack((real, imag), axis=0)
        return torch.tensor(stacked, dtype=torch.float32)

    def __getitem__(self, idx):
        x_i = self.ground_truth[idx]
        P_i = self.masks[idx]

        x_hat = self.get_x_hat(x_i, P_i)

        x_i = self.pad_to_target(x_i, self.target_size)
        x_hat = self.pad_to_target(x_hat, self.target_size)

        x_i_tensor = self.to_2channel_tensor(x_i)
        x_hat_tensor = self.to_2channel_tensor(x_hat)

        return x_i_tensor, x_hat_tensor


if __name__ == "__main__":
    file_path = "data/dataset.hdf5"
    dataset = MRIDataset(file_path)

    x_i, x_hat = dataset[0]

    print("x_i shape:", x_i.shape)
    print("x_hat shape:", x_hat.shape)

    x_i_array = x_i.numpy()
    x_hat_array = x_hat.numpy()

    x_i_abs = np.abs(x_i_array[0] + 1j * x_i_array[1])
    x_hat_abs = np.abs(x_hat_array[0] + 1j * x_hat_array[1])

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("x_hat")
    plt.imshow(x_hat_abs, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("x_i")
    plt.imshow(x_i_abs, cmap='gray')
    plt.show()
