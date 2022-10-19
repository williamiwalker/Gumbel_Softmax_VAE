import torch
import numpy as np
from torch.utils.data import Dataset


# reshape tensor
def reshape_fortran(x, shape):
    # Fortran/ Matlab like tensor  reshaping
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


# take MNIST data and give paired images for peer supervision
def rearrange_mnist(train_images, train_labels, num_factors,
                train_length=60000, sub_ids=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])):
    # Rearange MNIST dataset by grouping num_factors images of with identical labels together

    # Keep Only some digits
    num_digits = len(sub_ids)
    sub_samples_1 = torch.isin(train_labels, sub_ids)
    train_images = train_images[sub_samples_1]
    train_labels = train_labels[sub_samples_1]

    # Sub-Sample and shuffle original dataset
    perm = torch.randperm(len(train_images))
    train_images = train_images[perm[:train_length]]
    train_labels = train_labels[perm[:train_length]]

    # Dimension of each image
    image_size = train_images.shape[-1]

    # Minimum digit occurrence
    num_reps = torch.min(torch.sum(sub_ids.unsqueeze(dim=0) == train_labels.unsqueeze(dim=1), dim=0))
    num_reps = int((np.floor(num_reps / num_factors) * num_factors).squeeze().numpy())

    # Rearranged Datasets: num_reps x num_digits x image_size x image_size
    train_images_factors = torch.zeros((num_reps, num_digits, image_size, image_size))
    train_labels_factors = torch.zeros(num_reps, num_digits)
    for ii in range(len(sub_ids)):
        kept_images = (train_labels == sub_ids[ii])
        train_images_factors[:, ii, :, :] = train_images[kept_images.nonzero()[:num_reps]].squeeze()
        train_labels_factors[:, ii] = train_labels[kept_images.nonzero()[:num_reps]].squeeze()

    # Number of observation per digits
    num_obs_tmp = int(num_reps / num_factors)

    # Rearrange Datasets: num_obs_tmp x num_factors x num_digits x image_size x image_size
    train_images_factors.resize_(num_obs_tmp, num_factors, num_digits, image_size, image_size)
    train_labels_factors.resize_(num_obs_tmp, num_factors, num_digits)

    # Rearrange Datasets: num_obs x num_factors x image_size x image_size
    num_obs = num_obs_tmp * num_digits
    train_images_factors = torch.permute(train_images_factors, (0, 2, 1, 3, 4))
    train_labels_factors = torch.permute(train_labels_factors, (0, 2, 1))

    # train_images_factors.resize_(num_obs, num_factors, image_size, image_size)

    train_images_factors = reshape_fortran(train_images_factors, (num_obs, num_factors, image_size, image_size))
    train_labels_factors = reshape_fortran(train_labels_factors, (num_obs, num_factors))
    train_labels_factors = train_labels_factors[:, 0]

    # Use another Permutation to mix digits
    perm2 = torch.randperm(num_obs)
    train_images_factors = train_images_factors[perm2] # 27100 x 2 x 28 x 28
    train_labels_factors = train_labels_factors[perm2] # 27100

    # make image value range between 0 and 1
    max_val = torch.max(train_images_factors)

    observations = np.array([train_images_factors[:, ii].numpy() for ii in range(num_factors)])

    # Reshape Training Labels
    train_images_new = train_images_factors.reshape(num_obs * num_factors, image_size, image_size)
    train_labels_new = (train_labels_factors.unsqueeze(dim=1).repeat(1, num_factors)).reshape(
        num_obs * num_factors)

    return train_images_factors/max_val - 0.5, train_labels_factors


class PairedMNISTDataset(Dataset):
    """P pairs of different images of the same MNIST digit."""

    def __init__(self, dataset, labels, transform=None):

        self.transform = transform
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        sample = self.dataset[idx]

        if self.transform:
            sample = self.transform(self.dataset[idx])

        return (sample, self.labels[idx])


def save_checkpoint(model, optimizer, step, lossTrack, checkpoint_dir):
    checkpoint_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": lossTrack}
    checkpoint_path = checkpoint_dir+'learned_model.pth'# / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path))