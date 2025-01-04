import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, hyper_parameters):
        super(MLPModel, self).__init__()
        self.input_size = hyper_parameters['src_vocab_size']  # Length of the input encoded sequence
        self.embedding_size = hyper_parameters['d_model']  # Size of the embedding layer
        self.hidden_size = hyper_parameters['mlp_hidden']  # Size of the hidden layer
        self.output_size = hyper_parameters['tgt_vocab_size']  # Number of output categories
        self.num_layers = hyper_parameters['num_layers']  # Number of layers in the MLP
        # Define a multi-layer fully connected network
        self.embeddinglayer = nn.Embedding(self.input_size, self.embedding_size)  # Embedding Layer

        layers = []
        layers.append(nn.Linear(self.embedding_size, self.hidden_size))
        layers.append(nn.ReLU())

        for j in range(self.num_layers-1):  # Add hidden layers
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.hidden_size, self.output_size))  # output layer
        self.network = nn.Sequential(*layers)

    def forward(self, src):
        """
        src: (batch_size, seq_len)
        tgt: ignored in MLP since it's not sequence-to-sequence
        """

        assert src.dtype == torch.long, "Input src must be of type torch.LongTensor"
        assert src.max().item() < self.input_size, f"Input index exceeds embedding size {self.input_size}"
        assert src.min().item() >= 0, "Input index contains negative values"
        # The input to the MLP is `src`, and the output is the classification probabilities
        embedded = self.embeddinglayer(src)
        embeddedsrc = embedded.mean(dim=1)

        return self.network(embeddedsrc)
