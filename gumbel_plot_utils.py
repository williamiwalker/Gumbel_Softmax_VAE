'''
functions to plot MNIST outputs of vqvae
'''


import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from gumbel_utils import rearrange_mnist, PairedMNISTDataset
import numpy as np
import torch

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms, utils
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_loss(plot_folder, loss_tot):
    plt.figure()
    plt.plot(loss_tot)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig(plot_folder + 'loss.png')

def plot_VAE_reconst(saveFolder, name, images, outputs):
    # Feed a single batch through the model

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    images = images.reshape(64,56,28).unsqueeze(1)
    outputs = outputs.reshape(64,56,28).unsqueeze(1)

    image_grid = utils.make_grid(((images + 0.5) * 255).cpu().long())
    image_grid = image_grid.numpy()

    # print('image grid size',image_grid.shape,images.shape)

    ax1.set_title("Original", fontsize="large")
    ax1 = ax1.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='nearest')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    output_grid = utils.make_grid(outputs.cpu())
    output_grid = output_grid.numpy()

    # print('output shape',output_grid.shape, outputs.shape)

    ax2.set_title("Reconstructed", fontsize="large")
    ax2 = ax2.imshow(np.transpose(output_grid, (1, 2, 0)), interpolation='nearest')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.savefig(saveFolder + '/' + name + '.png')
    # fig.savefig(assets / "reconstructions.png", bbox_inches="tight", pad_inches = 0.5)


def digit_predictions(model, dataloader, args):

    model.eval()

    num_images = len(dataloader.dataset)
    digitPerdiction = np.zeros(num_images)
    trueLabels = np.zeros(num_images)
    posteriors = np.zeros((num_images, 10))
    #reconstructions = np.zeros((num_images, 2, 28, 28))

    curr_ind = 0
    for i, (images, labels) in enumerate(dataloader, 1):
        #images = images
        batchSize = images.shape[0]
        # print('image shape',images.shape,batchSize, curr_ind, curr_ind + batchSize, model.getDiscreteLatent(images).flatten().numpy().shape)
        recon_batch, qy = model(images, args.temp, args.hard)
        posteriors[curr_ind: curr_ind + batchSize] = qy.detach().cpu().numpy()
        digitPerdiction[curr_ind: curr_ind + batchSize] = torch.argmax(qy, dim=1).detach().flatten().cpu().numpy()
        trueLabels[curr_ind: curr_ind + batchSize] = labels.numpy()
        #reconstructions[curr_ind: curr_ind + batchSize] = recon_batch.detach().cpu().numpy()

        curr_ind = curr_ind + batchSize

    return digitPerdiction, trueLabels, posteriors







def permute_prediction(predictions, labels, used_digits):
    # Find Best Prediction in case the train labels are permuted

    # Number Of used Digits
    num_digits = len(used_digits)

    # Score Used to find Opt. Permutation
    scor_tot = np.eye(num_digits)

    for digit_id in range(num_digits):
        digit_cur = used_digits[digit_id]

        # Prediction digit = digit_cur
        idx = (predictions == digit_cur)

        # How many times 'digit_cur' from prediction is mapped to each digit in label.
        # scor_tot[digit_id, :] = (labels[idx].unsqueeze(dim=0) == used_digits.unsqueeze(dim=1)).sum(1)
        scor_tot[digit_id, :] = (np.expand_dims(labels[idx], axis=0) == np.expand_dims(used_digits, axis=1)).sum(1)


    # Maximise scor_tot using Hungarian Algorithm
    _, perm = linear_sum_assignment(-scor_tot)

    return torch.tensor(perm)


def plot_digit_recognition(saveFolder, name, model, dataloader):

    # acc_tot = torch.zeros(num_exp)

    # Deactivate dropouts
    model.eval()

    # Grasp Test set
    # test_images = test_data.test_data[torch.isin(test_data.test_labels, sub_ids)]
    # test_labels = test_data.test_labels[torch.isin(test_data.test_labels, sub_ids)]

    # Reduce training set
    # reduce_training_set = 20000

    # Convert Test datasets
    # test_tmp = torch.tensor(test_images.clone().detach(), dtype=torch.float32)

    # Use Recognition Network to classify digits
    predictions, labels, posteriors = digit_predictions(model, dataloader, model.args)
    # Get digit labels
    # labels = dataloader.labels.numpy()

    num_images = len(dataloader.dataset)
    sub_ids = np.unique(labels)

    # train_predictions = \
    #     torch.argmax(model.recognition_network.forward(train_images[:reduce_training_set].unsqueeze(dim=1)), dim=1)
    # test_predictions = \
    #     torch.argmax(model.recognition_network.forward(test_tmp.unsqueeze(dim=1)), dim=1)

    # Find best permutation between model clusters and digits identity
    perm_opt = permute_prediction(predictions, labels, sub_ids)

    # Permute Labels
    perm_predictions = perm_opt[predictions]

    print('len', len(perm_predictions), len(labels), num_images)

    # Train / Test performances
    accuracy = sum(abs(perm_predictions - labels) < 0.1) / num_images
    # test_accuracy = sum(abs(test_predictions - test_labels) < 0.01) / len(test_labels)

    # Summary
    results = str(np.round(accuracy.numpy(), 2))
    # test_results = str(np.round(test_accuracy.numpy(), 2))

    print('Training Accuracy = ' + str(accuracy))

    # acc_tot[exp_cur] = accuracy

    # print('Training Accuracy = ' + str(np.round(torch.mean(acc_tot).numpy(), 2))
    #       + ' +/- ' + str(np.round(torch.std(acc_tot).numpy(),2)))
    # print('Testing  Accuracy = ' + str(np.round(torch.mean(test_tot).numpy(), 2))
    #       + ' +/- ' + str(np.round(torch.std(test_tot).numpy(),2)))

    return accuracy


def plot_error_histogram(saveFolder, name, model, dataloader, args):

    # Evaluate model
    model.eval()

    # Use Recognition Network to classify digits
    predictions, labels, posteriors = digit_predictions(model, dataloader, args)

    num_images = len(dataloader.dataset)
    sub_ids = np.unique(labels)


    # Find best permutation between model clusters and digits identity
    perm_opt = permute_prediction(predictions, labels, sub_ids)

    # Permute Labels
    perm_predictions = perm_opt[predictions].detach().cpu().numpy()

    # Train / Test performances
    errors = (perm_predictions != labels)
    error_prediction = perm_predictions[errors]
    error_true = labels[errors]


    # Error matrix from true digit id to predicted error
    error_matrix = np.zeros((len(sub_ids), len(sub_ids)))
    for inde, digit_id in enumerate(error_true):
        # get predicted digit
        pred_id = error_prediction[inde]
        error_matrix[int(digit_id), int(pred_id)] += 1

    # plot error matrix
    fig = px.imshow(error_matrix.T, color_continuous_scale='ice',
                    labels=dict(x="true digit", y="predicted digit", color="number of errors"))
    fig.write_image(saveFolder + name + "error_matrix.png")

    # Error matrix from true digit id to predicted error
    confusion_matrix = np.zeros((len(sub_ids), len(sub_ids)))
    for inde, digit_id in enumerate(labels):
        # get predicted digit
        pred_id = perm_predictions[inde]
        confusion_matrix[int(digit_id), int(pred_id)] += 1

    # plot error matrix
    fig = px.imshow(confusion_matrix.T, color_continuous_scale='ice',
                    labels=dict(x="true digit", y="predicted digit", color="number of occurances"))
    fig.write_image(saveFolder + name + "confusion_matrix.png")

    # get sum of errors for each digit
    digit_errors = np.zeros(len(sub_ids))
    for indd in sub_ids:
        digit_errors[int(indd)] = np.sum(error_true == indd)

    # plot error histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=error_true))
    fig.update_layout(
        # title_text='Sampled Results',  # title of plot
        xaxis_title_text='digit',  # xaxis label
        yaxis_title_text='number of errors',  # yaxis label
        bargap=0.2  # gap between bars of adjacent location coordinates
    )
    fig.write_image(saveFolder + name + "error_histogram.png")


    # plot avg posterior entropy (VQVAE only uses one hot q(z|x))
    avg_posterior = np.mean(posteriors,axis=0)
    entropy_avg_posterior = -np.sum(avg_posterior * np.log(avg_posterior))

    return error_true, error_prediction, error_matrix, errors, entropy_avg_posterior


#
# def get_posteriors(saveFolder, name, model, ARGS, dataloader):
#     with torch.no_grad():
#         for exp_cur in range(len(ARGS)):
#             # get parameters for this run
#             paramDict = ARGS[str(exp_cur)]
#             # find the folder with the model
#             OUTPUT_FOLDER = paramDict['MAIN_FOLDER'] + '/' + paramDict['SUB_FOLDER']
#             saveFolder = parent_folder + OUTPUT_FOLDER + '/'
#
#             print('Gathering: ' + str(exp_cur + 1) + '/' + str(num_exp))
#
#             # name_file_cur = all_names[exp_cur]
#
#             # # Load Model
#             # with open('./../RPM_data/MNIST_all/' + name_file_cur, 'rb') as f:
#             #     model = pickle.load(f)
#             # Load model
#
#             # Init Model
#             model = UnstructuredRecognition(num_digits, observations, fit_params=fit_params)
#             # Load learned model
#             model.load_state_dict(loadModel('learned_model', saveFolder))
#
#             # Deactivate dropouts
#             model.recognition_network.eval()
#
#             # Grasp Test set
#             test_images = test_data.test_data[torch.isin(test_data.test_labels, sub_ids)]
#             test_labels = test_data.test_labels[torch.isin(test_data.test_labels, sub_ids)]
#
#             # Reduce training set
#             reduce_training_set = 20000
#
#             # Convert Test datasets
#             test_tmp = torch.tensor(test_images.clone().detach(), dtype=torch.float32)
#
#             # Use Recognition Network to classify digits
#             train_logits = model.recognition_network.forward(train_images[:reduce_training_set].unsqueeze(dim=1))
#             train_predictions = torch.argmax(train_logits, dim=1)
#
#             test_logits = model.recognition_network.forward(test_tmp.unsqueeze(dim=1))
#             test_predictions = torch.argmax(test_logits, dim=1)
#
#             # Calculate average entropy
#             train_dists = [Categorical(logits=logitsc) for logitsc in train_logits]
#             train_probs = torch.stack([dist.probs for dist in train_dists])
#
#             train_average_entropy[exp_cur] = -sum(
#                 [float(categorical_cross_entropy(dist, dist)) for dist in train_dists]) / float(len(train_dists))
#             train_average_posterior = Categorical(probs=torch.sum(train_probs, dim=0))
#             train_entropy_average_posterior[exp_cur] = -categorical_cross_entropy(train_average_posterior,
#                                                                                   train_average_posterior)
#             print('train_average ent, train_entropy avg post ', train_average_entropy[exp_cur],
#                   train_entropy_average_posterior[exp_cur])
#
#             test_dists = [Categorical(logits=logitsc) for logitsc in test_logits]
#             test_probs = torch.stack([dist.probs for dist in test_dists])
#
#             test_average_entropy[exp_cur] = -sum(
#                 [float(categorical_cross_entropy(dist, dist)) for dist in test_dists]) / float(len(test_dists))
#             test_average_posterior = Categorical(probs=torch.sum(test_probs, dim=0))
#             test_entropy_average_posterior[exp_cur] = -categorical_cross_entropy(test_average_posterior,
#                                                                                  test_average_posterior)
#             print('test_average ent, test_entropy avg post ', test_average_entropy[exp_cur],
#                   test_entropy_average_posterior[exp_cur])
#
#             # Factors Distribution
#             # self.factor_indpt = [Categorical(logits=logitsc) for logitsc in train_probs]
#
#             # Find best permutation between model clusters and digits identity
#             perm_opt = model.permute_prediction(train_predictions, train_labels[:reduce_training_set], sub_ids)
#
#             # Permute Labels
#             train_predictions = perm_opt[train_predictions]
#             test_predictions = perm_opt[test_predictions]
#
#             # Train / Test performances
#             train_accuracy = sum(
#                 abs(train_predictions - train_labels[:reduce_training_set]) < 0.1) / reduce_training_set
#             test_accuracy = sum(abs(test_predictions - test_labels) < 0.01) / len(test_labels)
#
#             # Summary
#             train_results = str(np.round(train_accuracy.numpy(), 2))
#             test_results = str(np.round(test_accuracy.numpy(), 2))
#
#             test_tot[exp_cur] = test_accuracy
#             trai_tot[exp_cur] = train_accuracy