from torch.utils.data import Dataset, DataLoader
import copy
import numpy as np

class RandDataset(Dataset):
    def __init__(self, p, index_list, all_data, all_label):
        super().__init__()
        self.p = p
        self.index_list = index_list
        self.all_data = all_data
        self.all_label = all_label

    def __len__(self):
        return len(self.index_list)
    
    def __getitem__(self, index):
        return self.all_data[self.index_list[index]], self.all_label[self.index_list[index]]
    
def get_all_data(p:int):
    '''
    given a prime number p, get all data and labels.

    input:
        p: a prime number
    
    output:
        all_data, all_label
    '''

    all_data = [[i, p, j, p+1] for i in range(p) for j in range(p)]
    all_data = np.array(all_data)
    all_label = (all_data.sum(axis=1) - 1) % p

    return all_data, all_label

def get_random_index(l:int, alpha: float):
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

def get_dataloaders(p:int, training_data_fraction:float, batch_size:int):
    '''
    get the training dataloader and validation dataloader
    input:
        p: prime number
        training_data_fraction
        batch_size
    output:
        train_dataloader, test_dataloader
    '''

    all_data, all_label = get_all_data(p)
    train_indexes, validation_indexes = get_random_index(l=p*p, alpha=training_data_fraction)

    my_train_data = RandDataset(p, index_list=train_indexes, all_data=all_data, all_label=all_label)
    my_test_data = RandDataset(p, index_list=validation_indexes, all_data=all_data, all_label=all_label)
        
    train_dataloader = DataLoader(dataset=my_train_data, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=my_test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader