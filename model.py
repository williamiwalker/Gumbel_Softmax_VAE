# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor
import json
from pathlib import Path
import sys

from gumbel_utils import rearrange_mnist, PairedMNISTDataset, save_checkpoint



def sample_gumbel(shape, args, eps=1e-20):
    U = torch.rand(shape)
    if args.cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature, args):
    y = logits + sample_gumbel(logits.size(), args)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, args, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature, args)

    if not hard:
        return y.view(-1, args.latent_dim * args.categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, args.latent_dim * args.categorical_dim)


class Residual(nn.Module):
    def __init__(self, channels):
        super(Residual, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class VAE_gumbel(nn.Module):
    def __init__(self, args):
        super(VAE_gumbel, self).__init__()

        # self.fc1 = nn.Linear(784, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, latent_dim * categorical_dim)
        self.conv1 = nn.Conv2d(2, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(4 * 4 * 20, 50)
        self.fc2 = nn.Linear(50, args.latent_dim * args.categorical_dim)
        self.latent_dim = args.latent_dim
        self.categorical_dim = args.categorical_dim
        self.args = args

        # self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        # self.fc5 = nn.Linear(256, 512)
        # self.fc6 = nn.Linear(512, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        channels = 128

        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_dim * self.categorical_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(channels),
            Residual(channels),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 2, 1)
        )
        self.linear1 = nn.Linear(self.latent_dim * self.categorical_dim, self.latent_dim * self.categorical_dim * 7 * 7)  # input

    def encode(self, x):
        # This is the same network used in RPM recognition network

        a = F.relu(F.max_pool2d(self.conv1(x), 2))
        a = F.relu(F.max_pool2d(self.conv2(a), 2))
        a = a.view(-1, 4 * 4 * 20)
        y = F.relu(self.fc1(a))
        return self.relu(self.fc2(y))

        # h1 = self.relu(self.fc1(x))
        # h2 = self.relu(self.fc2(h1))
        # return self.relu(self.fc3(h2))

    def decode(self, z):
        # This is the same as VQVAE

        batchSize = z.shape[0]
        x = self.linear1(z.view((batchSize, -1)))
        x = self.decoder(x.view((batchSize, self.latent_dim * self.categorical_dim, 7, 7)))
        return self.sigmoid(x).view(batchSize, 2, 784)
        # # print('dencoder output',x.shape, self.latent_dim, self.embedding_dim, self.channels)
        # B, _, H, W = x.size()
        # x = x.view(B, 2, 256, H, W).permute(0, 1, 3, 4, 2)
        # dist = Categorical(logits=x)

        # h4 = self.relu(self.fc4(z))
        # h5 = self.relu(self.fc5(h4))
        # return self.sigmoid(self.fc6(h5))

    def forward(self, x, temp, hard):
        q = self.encode(x)
        q_y = q.view(q.size(0), self.latent_dim, self.categorical_dim)
        z = gumbel_softmax(q_y, temp, self.args, hard)
        return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size())


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, categorical_dim):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 2, 784), size_average=False) / x.shape[0]

    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()

    return BCE + KLD


def train(epoch, args, model, optimizer, train_loader):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003
    model.train()
    train_loss = 0
    temp = args.temp
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, qy = model(data, temp, args.hard)
        loss = loss_function(recon_batch, data, qy, args.categorical_dim)
        loss.backward()
        train_loss += loss.item() * len(data)
        optimizer.step()
        if batch_idx % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)


def test(epoch, args, model, test_loader, saveFolder):
    temp_min = 0.5
    ANNEAL_RATE = 0.00003

    model.eval()
    test_loss = 0
    temp = args.temp
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        recon_batch, qy = model(data, temp, args.hard)
        test_loss += loss_function(recon_batch, data, qy, args.categorical_dim).item() * len(data)
        if i % 100 == 1:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * i), temp_min)
        if i == 0:
            n = min(data.size(0), 8)
            # comparison = torch.cat([data[:n,0],data[:n,1],
            #                         recon_batch.view(args.batch_size, 2, 28, 28)[:n, 0],
            #                         recon_batch.view(args.batch_size, 2, 28, 28)[:n, 1]])
            comparison = torch.cat([data[:n, 0].unsqueeze(1),
                                    recon_batch.view(args.batch_size, 2, 28, 28)[:n, 0].unsqueeze(1)])
            save_image(comparison.data.cpu(),
                       saveFolder + 'reconstruction_0.png', nrow=n)
            comparison = torch.cat([data[:n, 1].unsqueeze(1),
                                    recon_batch.view(args.batch_size, 2, 28, 28)[:n, 1].unsqueeze(1)])
            save_image(comparison.data.cpu(),
                       saveFolder + 'reconstruction_1.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def run(args, model, optimizer, saveFolder, train_loader, test_loader):
    lossTrack = np.zeros(args.epochs)
    for epoch in range(1, args.epochs + 1):
        lossTrack[epoch - 1] = train(epoch, args, model, optimizer, train_loader)
        test(epoch, args, model, test_loader, saveFolder)

        # This is for plotting the sample latent space
        # M = 64 * latent_dim
        # np_y = np.zeros((M, categorical_dim), dtype=np.float32)
        # np_y[range(M), np.random.choice(categorical_dim, M)] = 1
        # np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
        # sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
        # if args.cuda:
        #     sample = sample.cuda()
        # sample = model.decode(sample).cpu()[:,0,:,:]
        # save_image(sample.data.view(M // latent_dim, 1, 28, 28),
        #            saveFolder + 'sample.png')
    save_checkpoint(model, optimizer, lossTrack[-1], lossTrack, saveFolder)

