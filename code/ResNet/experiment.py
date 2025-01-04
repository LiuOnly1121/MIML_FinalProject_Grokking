import train
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

P = 47  # Prime number

HYPER_PARAMETERS = {
    'src_vocab_size': P + 2,    # 0, 1, ..., p-1, '+' (encoded as p), '=' (encoded as p+1)
    'tgt_vocab_size': P + 2,    # Same as above
    'max_seq_length': 4,        # Length of the input sequence
    'res_hidden': 128,          # Size of the hidden and embedding layers
    'num_layers': 4,            # Number of layers in the MLP
}

TRAINING_DATA_FRACTION_LIST = [0.3]
BATCH_SIZE = 80
EPOCHS = 100000

train.experiment_on_different_training_data_fraction(P, EPOCHS, TRAINING_DATA_FRACTION_LIST, HYPER_PARAMETERS, BATCH_SIZE, device)
