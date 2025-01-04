from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np

class SelectedDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
def get_all_data(p:int, n:int=2):
    '''
    given a prime number p, get all data and labels.

    input:
        p: a prime number
        n: x1 + x2 + ... + xn
    
    output:
        all_data, all_label
    '''

    all_data = np.zeros((p**n, 2*n), dtype=int)

    for i in range(n):
        all_data[:, 2*i] = np.tile(np.repeat(np.arange(p), p**(n-1-i)), p**i)
        all_data[:, 2*i + 1] = p if i < n-1 else p+1

    all_label = (all_data.sum(axis=1) - 1) % p

    return all_data, all_label

def get_one_sided_data(p:int, n:int=2):
    '''
    given a prime number p, get all one sided data and labels.
    x1 <= x2 <= ... <= xn

    input:
        p: a prime number
        n: x1 + x2 + ... + xn
    
    output:
        all_data, all_label
    '''

    all_data = np.zeros((p**n, 2*n), dtype=int)

    find_one_sided = np.ones((p**n,), dtype=int)

    for i in range(n):
        all_data[:, 2*i] = np.tile(np.repeat(np.arange(p), p**(n-1-i)), p**i)
        all_data[:, 2*i + 1] = p if i < n-1 else p+1
        if i > 0:
            find_one_sided = find_one_sided * np.where(all_data[:, 2*i] - all_data[:, 2*(i-1)] >= 0, 1, 0)

    all_data = all_data[np.where(find_one_sided == 1)]

    all_label = (all_data.sum(axis=1) - 1) % p

    return all_data, all_label

def get_random_index(l:int, alpha:float):
    '''
    Randomly choose int(l*alpha) indexes from range(l). Put them in an array (for training) and the rest in another array (for validation)

    input:
        l: the length of all data
        alpha: training data fraction
    output:
        training_index_list, validation_index_list
    '''

    size = int(l * alpha)

    train_indexes = np.random.choice(l, size=size, replace=False)
    validation_indexes = [i for i in range(l) if i not in train_indexes]
    validation_indexes = np.array(validation_indexes)

    return train_indexes, validation_indexes

def get_dataloaders_from_all_data(p:int, n:int, training_data_fraction:float, batch_size:int, one_sided=False):
    '''
    get the training dataloader and validation dataloader
    input:
        p: prime number
        n: x1 + x2 + ... + xn
        training_data_fraction
        batch_size
    output:
        train_dataloader, test_dataloader
    '''

    if one_sided:
        all_data, all_label = get_one_sided_data(p, n)
    else:
        all_data, all_label = get_all_data(p, n)

    train_indexes, validation_indexes = get_random_index(l=len(all_label), alpha=training_data_fraction)
    
    train_data, train_label = all_data[train_indexes], all_label[train_indexes]
    validation_data, validation_label = all_data[validation_indexes], all_label[validation_indexes]

    my_train_data = SelectedDataset(train_data, train_label)
    my_test_data = SelectedDataset(validation_data, validation_label)
        
    train_dataloader = DataLoader(dataset=my_train_data, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=my_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

# def get_dataloaders_from_random_data(p:int, n:int, training_data_size:int, test_data_size:int, batch_size:int):
#     '''
#     get the training dataloader and validation dataloader
#     input:
#         p: prime number
#         n: x1 + x2 + ... + xn
#         training_data_fraction
#         batch_size
#     output:
#         train_dataloader, test_dataloader
#     '''

#     total = training_data_size + test_data_size

#     total_range = p ** n

#     index = np.random.choice(np.arange(total_range), size=total, replace=False)

#     all_data = np.zeros((total, 2*n),dtype=int)

#     for i in range(n):
#         all_data[:, 2*i] = (index // (p ** (n - 1 - i))) % p
#         all_data[:, 2*i + 1] = p if i < n-1 else p+1

#     all_label = (all_data.sum(axis=1) - 1) % p

#     train_data, train_label = all_data[0:training_data_size-1,:], all_label[0:training_data_size-1]
#     validation_data, validation_label = all_data[training_data_size:total-1,:], all_label[training_data_size:total-1]

#     my_train_data = SelectedDataset(train_data, train_label)
#     my_test_data = SelectedDataset(validation_data, validation_label)
        
#     train_dataloader = DataLoader(dataset=my_train_data, batch_size=batch_size)
#     test_dataloader = DataLoader(dataset=my_test_data, batch_size=batch_size)

#     return train_dataloader, test_dataloader