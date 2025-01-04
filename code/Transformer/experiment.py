import train
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

P = 97      # the prime number

HYPER_PARAMETERS = {
    'src_vocab_size': P + 2,    # 0,1,...,p-1,'+'(encoded as p),'='(encoded as p+1)
    'tgt_vocab_size': P + 2,    # the same as src_vocab_size
    'max_seq_length': 4,
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 2,
    'd_ff': 512,
    'dropout': 0.1,
    'positional_encoding': True
    }

TRAINING_DATA_FRACTION_LIST = [0.3]  # You can train different models sequentially at the same time

BATCH_SIZE = 256

EPOCHS = 5000

train.experiment_on_different_training_data_fraction(P, EPOCHS, TRAINING_DATA_FRACTION_LIST, HYPER_PARAMETERS, BATCH_SIZE, device)