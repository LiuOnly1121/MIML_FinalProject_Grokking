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

P_2 = 31
N_LIST = [3]

BATCH_SIZE = 1024

EPOCHS_1 = 50000
EPOCHS_2 = 5000
EPOCHS_3 = 5000

SAVEPATH_1 = './result/result_onesided.csv'
FIGPATH_1 = './result/figures/result_onesided.png'

SAVEPATH_2 = './result/result_bothside.csv'
FIGPATH_2 = './result/figures/result_bothside.png'

SAVEPATH_3 = './result/result_onesided_n.csv'
FIGPATH_3 = './result/figures/result_onesided_n=3.png'

SAVEPATH_4 = './result/result_bothside_n.csv'
FIGPATH_4 = './result/figures/result_bothside_n=3.png'

train.experiment_on_different_training_data_fraction(P, EPOCHS_1, TRAINING_DATA_FRACTION_LIST, HYPER_PARAMETERS, BATCH_SIZE, device, savepath=SAVEPATH_1, n=N, one_sided=True)

train.experiment_on_different_training_data_fraction(P, EPOCHS_2, TRAINING_DATA_FRACTION_LIST, HYPER_PARAMETERS, BATCH_SIZE, device, savepath=SAVEPATH_2, n=N, one_sided=False)

visualization.visualization_1([SAVEPATH_1, SAVEPATH_2], TRAINING_DATA_FRACTION_LIST + TRAINING_DATA_FRACTION_LIST, FIGPATH_2, scale=0.4, cut=True, sharextitle=True, sharex=True, column=3)

train.experiment_on_different_n(P_2, EPOCHS_3, N_LIST, HYPER_PARAMETERS, BATCH_SIZE, device, csv_path=SAVEPATH_3, one_sided=True, training_data_fraction=0.3)

train.experiment_on_different_n(P_2, EPOCHS_3, N_LIST, HYPER_PARAMETERS, BATCH_SIZE, device, csv_path=SAVEPATH_4, one_sided=False, training_data_fraction=0.3)

visualization.visualization_2([SAVEPATH_3, SAVEPATH_4], N_LIST + N_LIST, FIGPATH_4, scale=0.2, cut=False, column=1, sharextitle=True, sharex=True)