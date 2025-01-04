import train
import torch
import numpy as np
import random
import visualization

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

P = 17      # the prime number

TRAINING_DATA_FRACTION = 0.5

N_LIST = [2,3,4]

HYPER_PARAMETERS = {
    'src_vocab_size': P + 2,    # 0,1,...,p-1,'+'(encoded as p),'='(encoded as p+1)
    'tgt_vocab_size': P + 2,    # the same as src_vocab_size
    'max_seq_length': 2 * max(N_LIST),        # 'x1 + x2 + ... + xn ='
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 2,
    'd_ff': 512,
    'dropout': 0.1,
    'positional_encoding': True
    }

BATCH_SIZE = 1024

EPOCHS = 500

CSV_PATH = './result/result_n=2,3,4.csv'
FIG_PATH = './result/figures/result_n=2,3,4.png'

# train.experiment_on_different_n(P, EPOCHS, N_LIST, HYPER_PARAMETERS, BATCH_SIZE, device, training_data_fraction=TRAINING_DATA_FRACTION, csv_path=CSV_PATH)

visualization.visualization_2(CSV_PATH, N_LIST, FIG_PATH)

P_2 = 31
TRAINING_DATA_FRACTION_2 = 0.5
N_LIST_2 = [2,3]

EPOCHS_2 = 5000

HYPER_PARAMETERS_2 = {
    'src_vocab_size': P_2 + 2,    # 0,1,...,p-1,'+'(encoded as p),'='(encoded as p+1)
    'tgt_vocab_size': P_2 + 2,    # the same as src_vocab_size
    'max_seq_length': 2 * max(N_LIST_2),        # 'x1 + x2 + ... + xn ='
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 2,
    'd_ff': 512,
    'dropout': 0.1,
    'positional_encoding': True
    }

CSV_PATH_2 = './result/result_n=2,3.csv'
FIG_PATH_2 = './result/figures/result_n=2,3.png'

# train.experiment_on_different_n(P_2, EPOCHS_2, N_LIST_2, HYPER_PARAMETERS_2, BATCH_SIZE, device, training_data_fraction=TRAINING_DATA_FRACTION_2, csv_path=CSV_PATH_2)

visualization.visualization_2(CSV_PATH_2, N_LIST_2, FIG_PATH_2, column=2, cut=True)
