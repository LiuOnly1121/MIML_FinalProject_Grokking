import train
import torch
import numpy as np
import visualization
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

P = 97      # the prime number
N = 2       # it is highly recommend that N <= 4

HYPER_PARAMETERS = {
    'src_vocab_size': P + 2,    # 0,1,...,p-1,'+'(encoded as p),'='(encoded as p+1)
    'tgt_vocab_size': P + 2,    # the same as src_vocab_size
    'max_seq_length': 2 * N,        # 'x1 + x2 + ... + xn ='
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 2,
    'd_ff': 512,
    'dropout': 0.1,
    'positional_encoding': True,
    }

BATCH_SIZE = 1024

EPOCHS = 1000

def get_optimizer(model, i):

    return[
        torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9),
        torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9),
        torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.98), eps=1e-9),
        torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9),                        #dropout = 0.1
        torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=0.05, eps=1e-9),
        torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=1.0, eps=1e-9),
        torch.optim.RMSprop(model.parameters(), lr=5e-4),
        torch.optim.RMSprop(model.parameters(), lr=1e-3),
        torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    ][i]

OPTIMIZER_NAMES = [
    'adam lr=0.0001',
    'adam lr=0.001',
    'adam lr=0.003',
    'adam lr=0.0001, dropout=0.1',
    'adamW weight decay=0.05',
    'adamW weight decay=1.0',
    'RMSprop lr=0.0005',
    'RMSprop lr=0.001',
    'SGD momentum=0.9',
]

DROPOUT_LIST = [0, 0, 0, 0.1, 0, 0, 0, 0, 0]

TRAINING_DATA_FRACTION_LIST = np.linspace(0.3, 0.8, 11).tolist()

REPEAT_TIME = 3

CSV_PATH = './result/result_opt.csv'
FIG_PATH = './result/figures/result_opt_test.png'

train.experiment_on_different_optimizers(P, get_optimizer, DROPOUT_LIST, OPTIMIZER_NAMES, EPOCHS, TRAINING_DATA_FRACTION_LIST, HYPER_PARAMETERS,BATCH_SIZE, device, N, REPEAT_TIME,CSV_PATH)

visualization.visualization_3(CSV_PATH, OPTIMIZER_NAMES, TRAINING_DATA_FRACTION_LIST, fig_path=FIG_PATH)