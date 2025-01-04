import torch
import torch.nn as nn


class MLPResNet(nn.Module):
    def __init__(self, hyper_parameters):
        super(MLPResNet, self).__init__()
        self.input_size = hyper_parameters['src_vocab_size']  # Length of the input encoded sequence
        self.hidden_size = hyper_parameters['res_hidden']  # Size of the hidden layer
        self.output_size = hyper_parameters['tgt_vocab_size']  # Number of output categories
        self.num_layers = hyper_parameters['num_layers']  # Number of layers in the MLP

        # Define a multi-layer fully connected network
        self.embeddinglayer = nn.Embedding(self.input_size, self.hidden_size)  # Embedding Layer
        self.hiddenlayer = nn.Linear(self.hidden_size, self.hidden_size)  # Hidden Layer
        self.relu = nn.ReLU(inplace=False)  # Avoid in-place operation
        self.outputlayer = nn.Linear(self.hidden_size, self.output_size)  # Output Layer

    def forward(self, src):
        # Check if the input is valid
        assert src.dtype == torch.long, "Input src must be of type torch.LongTensor"
        assert src.max().item() < self.input_size, f"Input index exceeds embedding size {self.input_size}"
        assert src.min().item() >= 0, "Input index contains negative values"

        # Embedding layer: Map input indices to vectors
        embedded = self.embeddinglayer(src)  # (batch_size, seq_len, hidden_size)
        out = embedded.mean(dim=1)  # Take the mean along the sequence length dimension, (batch_size, hidden_size)

        # Initialize residual connection
        residual = out.clone()  # Create a copy for residual connection

        # Multi-layer fully connected network
        for j in range(self.num_layers):
            # Hidden layer + ReLU
            out = self.hiddenlayer(out)
            out = self.relu(out)  # Activation

            # Residual connection
            out = out + residual  # Add residual
            out = self.relu(out)  # Activate again

        # Output layer
        out = self.outputlayer(out)  # (batch_size, output_size)
        return out

