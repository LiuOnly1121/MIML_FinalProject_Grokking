import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
from transformer import Transformer
import matplotlib.pyplot as plt
import math
import time
import os

def train(model, train_dataloader, optimizer, criterion, device):
    '''
    given the model, do one epoch of training, and return loss and acc
    '''
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for _, (data, target) in enumerate(train_dataloader):

        data = data.to(device)
        target = target.to(device)

        data0 = torch.zeros_like(data)
        data0.copy_(data)
        data0 = data0.to(device)
        
        y_pred = model.forward(data, data0)
 
        optimizer.zero_grad()
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()

        train_loss += criterion(y_pred, target).detach().to('cpu').numpy()
        _, predicted = torch.max(y_pred, 1)
        total += len(target)
        correct += torch.eq(predicted, target).sum().detach().to('cpu').numpy()

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
        for _, (data, target) in enumerate(test_dataloader):

            data = data.to(device)
            target = target.to(device)

            data0 = torch.zeros_like(data)
            data0.copy_(data)
            data0 = data0.to(device)

            y_pred = model(data, data0)

            test_loss += criterion(y_pred, target).detach().to('cpu').numpy()
            _, predicted = torch.max(y_pred, 1)
            total += len(target)
            correct += torch.eq(predicted, target).sum().detach().to('cpu').numpy()

    return test_loss / len(test_dataloader), 100. * correct / total

# def visualization(train_acc_list, test_acc_list, train_loss_list, test_loss_list, savepath, state, x_ticks=None, save=True):

#     plt.figure()
#     plt.xscale('symlog')
#     try:
#         x_ticks = x_ticks[x_ticks <= len(train_acc_list)]
#         plt.xticks(x_ticks)
#     except:
#         pass
#     plt.xlim(left=1)
#     plt.plot(train_loss_list, label='train loss')
#     plt.plot(test_loss_list, label='validation loss')
#     plt.legend()
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True, axis='y')
#     if save:
#         plt.savefig(os.path.join(savepath, 'loss_' + state + '.png'), bbox_inches='tight')
#     plt.close()

#     plt.figure()
#     plt.xscale('symlog')
#     try:
#         x_ticks = x_ticks[x_ticks <= len(train_acc_list)]
#         plt.xticks(x_ticks)
#     except:
#         pass
#     plt.xlim(left=1)
#     plt.plot(train_acc_list, label='train acc')
#     plt.plot(test_acc_list, label='validation acc')
#     plt.legend()
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy(%)')
#     plt.grid(True, axis='y')
#     if save:
#         plt.savefig(os.path.join(savepath, 'acc_' + state + '.png'), bbox_inches='tight')
#     plt.close()


def train_and_validation(model, epochs, train_dataloader, test_dataloader, optimizer, criterion, path, state, device):
    '''
    train the model, and record the loss and accuracy on training data and validation data for each epoch
    '''
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_dataloader, criterion, device)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,  Validation Loss: {test_loss:.4f}, Validation Acc: {test_acc:.2f}%")
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

    train_acc_list = np.array(train_acc_list)
    test_acc_list = np.array(test_acc_list)
    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)

    x_ticks = np.concatenate([(np.arange(9) + 2) * 10**i for i in range(math.ceil(math.log10(epochs)))])
    x_ticks = np.concatenate([np.array([1]), x_ticks])

    # visualization
    # fig_path = os.path.join(path, 'result/figures')
    # visualization(train_acc_list, test_acc_list, train_loss_list, test_loss_list, fig_path, state, x_ticks=x_ticks, save=save)

    # save the model

    torch.save(model.state_dict(), os.path.join(path, 'result/models/model_' + state + '.pth'))

    return [train_acc_list, test_acc_list, train_loss_list, test_loss_list]


def experiment_on_different_training_data_fraction(p, epochs, training_data_fraction_list, hyper_parameters, batch_size, device, n=2, savepath='./result/result_alpha.csv', one_sided=False):

    current_path = os.getcwd()

    print('*' * 100)
    print('we will train {} models'.format(len(training_data_fraction_list)))
    print('p = {} , n = {}'.format(p, n))
    print('training data fractions will be:', training_data_fraction_list)
    print('*' * 100)

    multi_indexes = pd.MultiIndex.from_product([training_data_fraction_list, ['train acc', 'validation acc', 'train loss', 'validation loss']], names=['training data fraction', 'process'])
    columns = np.arange(epochs) + 1
    train_states = []

    for training_data_fraction in training_data_fraction_list:

        transformer = Transformer(hyper_parameters)
        transformer.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(transformer.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=1.0, eps=1e-9)
        train_dataloader, validation_dataloader = dataset.get_dataloaders_from_all_data(p, n, training_data_fraction, batch_size, one_sided=one_sided)

        print('=' * 100)
        print('the model with [ p = {}, training data fraction = {} ]'.format(p, training_data_fraction))
        print('-' * 42 + ' training start ' + '-' * 42)

        state = 'p={}_alpha={}_n={}'.format(p,training_data_fraction,n)

        start_time = time.time()

        train_states.extend(train_and_validation(transformer, epochs, train_dataloader, validation_dataloader, optimizer, criterion, current_path, state, device))

        end_time = time.time()

        print('-' * 36 + ' training time: %02d min %02d s ' % (int(end_time - start_time) // 60, int(end_time - start_time) % 60) + '-' * 36)

    train_states = pd.DataFrame(train_states, index=multi_indexes, columns=columns)
    train_states.to_csv(savepath)

    print('=' * 100)

    print('the training of all models has finished!')


def experiment_on_different_n(p, epochs, n_list, hyper_parameters, batch_size, device, training_data_fraction=0.5, csv_path='./result/result_2.csv', one_sided=False):

    current_path = os.getcwd()

    print('*' * 100)
    print('we will train {} models'.format(len(n_list)))
    print('p = {} , alpha = {}'.format(p, training_data_fraction))
    print('n will be:', n_list)
    print('*' * 100)

    multi_indexes = pd.MultiIndex.from_product([n_list, ['train acc', 'validation acc', 'train loss', 'validation loss']], names=['n', 'process'])
    columns = np.arange(epochs) + 1
    train_states = []

    for n in n_list:

        hyper_parameters['max_seq_length'] = 2 * n

        transformer = Transformer(hyper_parameters)
        transformer.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.AdamW(transformer.parameters(), lr=7e-4, betas=(0.9, 0.98), weight_decay=1.0, eps=1e-9)

        train_dataloader, validation_dataloader = dataset.get_dataloaders_from_all_data(p, n, training_data_fraction, batch_size, one_sided=one_sided)

        print('=' * 100)
        print('the model with [ p = {}, n = {} ]'.format(p, n))
        print('-' * 42 + ' training start ' + '-' * 42)

        state = 'p={}_alpha={}_n={}'.format(p,training_data_fraction,n)

        start_time = time.time()

        train_states.extend(train_and_validation(transformer, epochs, train_dataloader, validation_dataloader, optimizer, criterion, current_path, state, device))

        end_time = time.time()

        print('-' * 36 + ' training time: %02d min %02d s ' % (int(end_time - start_time) // 60, int(end_time - start_time) % 60) + '-' * 36)

    train_states = pd.DataFrame(train_states, index=multi_indexes, columns=columns)
    train_states.to_csv(csv_path)

    print('=' * 100)

    print('the training of all models has finished!')

def experiment_on_different_optimizers(p, get_optimizer, dropout_list, optimizer_names, epochs, training_data_fraction_list, hyper_parameters, batch_size, device, n=2, repeat_time=10, csv_path='./result/result_3.csv'):

    current_path = os.getcwd()

    print('*' * 100)
    print('we will train {} models'.format(len(training_data_fraction_list)))
    print('p = {} , n = {}'.format(p, n))
    print('training data fractions will be:', training_data_fraction_list)
    print('*' * 100)

    states = []
    for i in range(len(optimizer_names)):

        states.append('p={} opt:{}_dropout={}'.format(p,optimizer_names[i],dropout_list[i]))

    mult_indexes = pd.MultiIndex.from_product([states, training_data_fraction_list], names=['state', 'alpha'])
    columns = (np.arange(repeat_time) + 1).tolist()
    columns.append('average')
    max_acc_list = []

    for i in range(len(optimizer_names)):
    
        hyper_parameters['dropout'] = dropout_list[i]

        state = states[i]

        for j in range(len(training_data_fraction_list)):

            training_data_fraction = training_data_fraction_list[j]
            temp_acc = []

            # print('=' * 100)
            # print('the model with [ p = {}, training data fraction = {} ]'.format(p, training_data_fraction))
            print('-' * 42 + ' training start ' + '-' * 42)

            for r in range(repeat_time):

                print('optimizer: {} / {} || alpha: {} / {} || progress: {} / {}'.format(i+1, len(optimizer_names), j+1, len(training_data_fraction_list), r+1, repeat_time))

                transformer = Transformer(hyper_parameters)
                transformer.to(device)

                criterion = nn.CrossEntropyLoss().to(device)
                # optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
                optimizer = get_optimizer(transformer, i)
                train_dataloader, validation_dataloader = dataset.get_dataloaders_from_all_data(p, n, training_data_fraction, batch_size)

                start_time = time.time()

                validation_acc = train_and_validation(transformer, epochs, train_dataloader, validation_dataloader, optimizer, criterion, current_path, state, device)[1]
                max_acc = validation_acc.max()
                temp_acc.append(max_acc)

                end_time = time.time()

                print('-' * 36 + ' training time: %02d min %02d s ' % (int(end_time - start_time) // 60, int(end_time - start_time) % 60) + '-' * 36)

            # temp_acc = np.array(temp_acc)

            max_acc_list.append(temp_acc)

    max_acc_np = np.array(max_acc_list)
    average_acc = max_acc_np.mean(axis=1)
    max_acc_np = np.column_stack((max_acc_np, average_acc))

    max_acc_df = pd.DataFrame(max_acc_np, index=mult_indexes, columns=columns)

    max_acc_df.to_csv(csv_path)

    print('=' * 100)

    print('the training of all models has finished!')

    return(max_acc_list)

# def experiment_on_more_optimizers(p, get_optimizer, dropout_list, optimizer_names, epochs, training_data_fraction_list, hyper_parameters, batch_size, device, old_csv, n=2, repeat_time=10, csv_path='./result/result_opt.csv'):

#     current_path = os.getcwd()

#     print('*' * 100)
#     print('we will train {} models'.format(len(training_data_fraction_list)))
#     print('p = {} , n = {}'.format(p, n))
#     print('training data fractions will be:', training_data_fraction_list)
#     print('*' * 100)

#     states = []
#     for i in range(len(optimizer_names)):

#         states.append('p={} opt:{}_dropout={}'.format(p,optimizer_names[i],dropout_list[i]))

#     mult_indexes = pd.MultiIndex.from_product([states, training_data_fraction_list], names=['state', 'alpha'])
#     columns = (np.arange(repeat_time) + 1).tolist()
#     columns.append('average')
#     max_acc_list = []

#     for i in range(len(optimizer_names)):
    
#         hyper_parameters['dropout'] = dropout_list[i]

#         state = states[i]

#         for j in range(len(training_data_fraction_list)):

#             training_data_fraction = training_data_fraction_list[j]
#             temp_acc = []

#             # print('=' * 100)
#             # print('the model with [ p = {}, training data fraction = {} ]'.format(p, training_data_fraction))
#             print('-' * 42 + ' training start ' + '-' * 42)

#             for r in range(repeat_time):

#                 print('optimizer: {} / {} || alpha: {} / {} || progress: {} / {}'.format(i+1, len(optimizer_names), j+1, len(training_data_fraction_list), r+1, repeat_time))

#                 transformer = Transformer(hyper_parameters)
#                 transformer.to(device)

#                 criterion = nn.CrossEntropyLoss().to(device)
#                 # optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
#                 optimizer = get_optimizer(transformer, i)
#                 train_dataloader, validation_dataloader = dataset.get_dataloaders_from_all_data(p, n, training_data_fraction, batch_size)

#                 start_time = time.time()

#                 validation_acc = train_and_validation(transformer, epochs, train_dataloader, validation_dataloader, optimizer, criterion, current_path, state, device)[1]
#                 max_acc = validation_acc.max()
#                 temp_acc.append(max_acc)

#                 end_time = time.time()

#                 print('-' * 36 + ' training time: %02d min %02d s ' % (int(end_time - start_time) // 60, int(end_time - start_time) % 60) + '-' * 36)

#             # temp_acc = np.array(temp_acc)

#             max_acc_list.append(temp_acc)

#     max_acc_np = np.array(max_acc_list)
#     average_acc = max_acc_np.mean(axis=1)
#     max_acc_np = np.column_stack((max_acc_np, average_acc))

#     max_acc_df = pd.DataFrame(max_acc_np, index=mult_indexes, columns=columns)

#     original_acc_df = pd.read_csv(old_csv)

#     max_acc_df.to_csv(csv_path)
#     max_acc_df = pd.read_csv(csv_path)

#     new_acc_df = pd.concat([original_acc_df, max_acc_df])

#     print(new_acc_df)

#     new_acc_df.to_csv(csv_path)

#     print('=' * 100)

#     print('the training of all models has finished!')

#     return(max_acc_list)