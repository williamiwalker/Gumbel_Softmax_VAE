'''
plot everything for GSSOFT and VQVAE
'''

from model import VAE_gumbel
import sys, os
import json
import torch
import argparse
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms, utils
from gumbel_utils import rearrange_mnist, PairedMNISTDataset
from torch.utils.data import DataLoader
from gumbel_plot_utils import plot_VAE_reconst, plot_digit_recognition, plot_error_histogram, plot_loss
import pickle


# just check if cuda is available
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")


# SLURM_ARRAY_TASK_ID = sys.argv[1]
# print('SLURM_ARRAY_TASK_ID ',SLURM_ARRAY_TASK_ID)

ARG_FILE_NAME = 'arguments_gumbel_softmax_2.json'
parent_folder = '/nfs/gatsbystor/williamw/gprpm_plots/'
#parent_folder = '/home/william/mnt/gatsbystor/gprpm_plots/'
ARGUMENT_FILE = parent_folder + 'arg_files/' + ARG_FILE_NAME

with open(ARGUMENT_FILE) as json_file:
    ARGS = json.load(json_file)
    print('PARAMETERS ', ARGS[str(0)])
    paramDict = ARGS[str(0)]
# make arguments from dict to namespace
args = argparse.Namespace(**paramDict)
######################################################################################
# INPUT MNIST DATASET
######################################################################################

# get MNIST data
# Load MNIST
train_data = datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    transform=ToTensor()
)

# Random seeds
torch.manual_seed(1)

# Number of Conditionally independent Factors
num_factors = 2

# Sub-Sample original dataset
train_length = 60000
test_length = 10000

# Keep Only some digits
sub_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
num_digits = len(sub_ids)

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
training_data, train_labels = rearrange_mnist(
    train_data.train_data, train_data.train_labels, num_factors, train_length=train_length, sub_ids=sub_ids)
training_dataset = PairedMNISTDataset(training_data, train_labels)

# Rearrange MNIST by grouping num_factors Conditionally independent Observations together
test_data, test_labels = rearrange_mnist(
    test_data.test_data, test_data.test_labels, num_factors, train_length=test_length, sub_ids=sub_ids)
test_dataset = PairedMNISTDataset(test_data, test_labels)

training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=True)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True,
                             num_workers=args.num_workers, pin_memory=True)

#####################################################################################



for indr in range(len(ARGS)):

    with open(ARGUMENT_FILE) as json_file:
        ARGS = json.load(json_file)
        print('PARAMETERS ',ARGS[str(indr)])
        paramDict = ARGS[str(indr)]


    OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/' + paramDict['SUB_FOLDER']
    saveFolder = parent_folder + OUTPUT_FOLDER + '/'

    # make arguments from dict to namespace
    args = argparse.Namespace(**paramDict)
    args.cuda = torch.cuda.is_available()

    model = VAE_gumbel(args)

    # # create model
    # if args.model == "VQVAE":
    #     model = VQVAE(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)
    # if args.model == "GSSOFT":
    #     model = GSSOFT(args.channels, args.latent_dim, args.num_embeddings, args.embedding_dim)

    # load learned model
    checkpoint_path = saveFolder + 'learned_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    lossTrack = checkpoint['loss']

    # plot loss
    plot_loss(saveFolder, lossTrack)


    # get images and reconstructions for plotting VAE reconstructions
    model.eval()
    for images, _ in test_dataloader:
        # images = images.to(device)
        with torch.no_grad():
            recon_batch, qy = model(images, args.temp, args.hard)
            # dist, _, _ = model(images)
            # outputs = dist.probs.argmax(dim=-1)
        break

    train_error_true, train_error_prediction, train_error_matrix, train_errors, train_entropy_avg_posterior = \
        plot_error_histogram(saveFolder, 'train_', model, training_dataloader, args)
    test_error_true, test_error_prediction, test_error_matrix, test_errors, test_entropy_avg_posterior = plot_error_histogram(
        saveFolder, 'test_', model, training_dataloader, args)
    plot_VAE_reconst(saveFolder, 'reconstructions', images, recon_batch.reshape(recon_batch.shape[0],2, 28, 28))

    # plot_digit_recognition(saveFolder, 'digit_recognition', model, test_dataloader)
    train_accuracy = plot_digit_recognition(saveFolder, 'digit_recognition', model, training_dataloader)
    test_accuracy  = plot_digit_recognition(saveFolder, 'digit_recognition', model, test_dataloader)

    print('train accuracy', train_accuracy)
    print('test accuracy', test_accuracy)

    # save train and test accuracy in json
    accuracy = {'train_accuracy':float(train_accuracy), 'test_accuracy':float(test_accuracy),
                'train_error_true': train_error_true, 'train_error_prediction':train_error_prediction, 'train_error_matrix':train_error_matrix,
                'train_errors':train_errors, 'train_entropy_avg_posterior':train_entropy_avg_posterior,
                'test_error_true': test_error_true, 'test_error_prediction': test_error_prediction,
                'test_error_matrix': test_error_matrix,
                'test_errors': test_errors, 'test_entropy_avg_posterior': test_entropy_avg_posterior
                }
    accuracy_file = saveFolder + 'accuracy.pkl'
    with open(accuracy_file, 'wb') as handle:
        pickle.dump(accuracy, handle)
    # with open(accuracy_file, 'w') as f:
    #     json.dump(accuracy, f, indent=4)