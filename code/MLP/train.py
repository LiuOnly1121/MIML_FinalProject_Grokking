import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from mlp import MLPModel
import matplotlib.pyplot as plt
import math
import time
import os
import numpy as np


def train(model, train_dataloader, optimizer, criterion, device):
    '''
    given the model, do one epoch of training, and return loss and acc
    '''
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch, (data, target) in enumerate(train_dataloader):
        # Convert data to float32 and move to the specified device
        data = data.to(device).long()
        target = target.to(device)

        # Forward propagation
        y_pred = model.forward(data)


        optimizer.zero_grad()
        loss = criterion(y_pred, target.long())
        loss.backward()
        optimizer.step()

        train_loss += criterion(y_pred, target.long()).detach().to('cpu').numpy()
        _, predicted = torch.max(y_pred, 1)
        total += len(target)
        correct += torch.eq(predicted, target.long()).sum().detach().to('cpu').numpy()

    return train_loss / len(train_dataloader), 100. * correct / total


def test(model, test_dataloader, criterion, device):
    '''
    given the current state of the model and a dataset, get the acc and loss
    '''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch, (data, target) in enumerate(test_dataloader):
            data = data.to(device).long()
            target = target.to(device)

            y_pred = model(data)

            test_loss += criterion(y_pred, target.long()).detach().to('cpu').numpy()
            _, predicted = torch.max(y_pred, 1)
            total += len(target)
            correct += torch.eq(predicted, target.long()).sum().detach().to('cpu').numpy()

    return test_loss / len(test_dataloader), 100. * correct / total


def train_and_validation(model, epochs, train_dataloader, test_dataloader, optimizer, criterion, path, p, alpha,
                         device):
    '''
    train the model, and record the loss and accuracy on training data and validation data for each epoch
    '''
    # Ensure the directory exists
    result_path = os.path.join(path, 'result/figures/')
    os.makedirs(result_path, exist_ok=True)  # Create the directory if it does not exist
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_dataloader, criterion, device)
        if (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,  Validation Loss: {test_loss:.4f}, Validation Acc: {test_acc:.2f}%")
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    train_acc_list = np.array(train_acc_list)
    test_acc_list = np.array(test_acc_list)
    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)

    X = np.arange(len(train_acc_list)) + 1

    x_ticks = np.concatenate([(np.arange(9) + 2) * 10 ** i for i in range(math.ceil(math.log10(epochs)))])
    x_ticks = np.concatenate([np.array([1]), x_ticks])

    # visualization
    plt.figure()
    plt.xscale('symlog')
    plt.xticks(x_ticks)
    plt.plot(X, train_loss_list, label='train loss')
    plt.plot(X, test_loss_list, label='validation loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(path, 'result/figures/loss_{}_{}.png').format(p, alpha), bbox_inches='tight')

    plt.figure()
    plt.xscale('symlog')
    plt.plot(X, train_acc_list, label='train acc')
    plt.plot(X, test_acc_list, label='validation acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.savefig(os.path.join(path, 'result/figures/acc_{}_{}.png'.format(p, alpha)), bbox_inches='tight')

    # Ensure the directory exists
    result_path = os.path.join(path, 'result/models/')
    os.makedirs(result_path, exist_ok=True)  # Create the directory if it does not exist

    # save the model
    torch.save(model.state_dict(), os.path.join(path, 'result/models/model_{}_{}.pth'.format(p, alpha)))

    return [train_acc_list, test_acc_list, train_loss_list, test_loss_list]


def experiment_on_different_training_data_fraction(p, epochs, training_data_fraction_list, hyper_parameters, batch_size, device):
    current_path = os.getcwd()

    print('*' * 100)
    print('we will train {} models'.format(len(training_data_fraction_list)))
    print('p = {}'.format(p))
    print('training data fractions will be:')
    print(training_data_fraction_list)
    print('*' * 100)

    multi_indexes = pd.MultiIndex.from_product(
        [training_data_fraction_list, ['train acc', 'validation acc', 'train loss', 'validation loss']],
        names=['training data fraction', 'process']
    )
    columns = np.arange(epochs) + 1
    train_states = []

    for training_data_fraction in training_data_fraction_list:
        # Initialize the MLP model
        model = MLPModel(hyper_parameters)
        model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=7e-4, betas=(0.96, 0.99), weight_decay=1)

        train_dataloader, validation_dataloader = dataset.get_dataloaders(p, training_data_fraction, batch_size)

        print('=' * 100)
        print('the model with [ p={}, training data fraction={} ]'.format(p, training_data_fraction))
        print('-' * 42 + ' training start ' + '-' * 42)

        start_time = time.time()

        train_states.extend(
            train_and_validation(
                model, epochs, train_dataloader, validation_dataloader, optimizer, criterion, current_path, p, training_data_fraction, device
            )
        )

        end_time = time.time()

        print('-' * 36 + ' training time: %02d min %02d s ' % (int(end_time - start_time) // 60, int(end_time - start_time) % 60) + '-' * 36)

    # Ensure the directory exists
    result_path = os.path.join(current_path, 'result/')
    os.makedirs(result_path, exist_ok=True)  # Create the directory if it does not exist

    train_states = pd.DataFrame(train_states, index=multi_indexes, columns=columns)
    train_states.to_csv('./result/result.csv')

    print('=' * 100)
    print('the training of all models has finished!')