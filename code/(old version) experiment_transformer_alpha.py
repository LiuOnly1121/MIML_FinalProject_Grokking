import train
import torch
import numpy as np
import random
import visualization

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# set_seed(42)

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
    'positional_encoding': True
    }

TRAINING_DATA_FRACTION_LIST = [0.3, 0.5, 0.7]

BATCH_SIZE = 1024

EPOCHS = 1000

CSV_PATH = './result/result_alpha.csv'
FIG_PATH = './result/figures/result_alpha.png'

train.experiment_on_different_training_data_fraction(P, EPOCHS, TRAINING_DATA_FRACTION_LIST, HYPER_PARAMETERS, BATCH_SIZE, device, N, savepath=CSV_PATH)

visualization.visualization_1(CSV_PATH, TRAINING_DATA_FRACTION_LIST, FIG_PATH)