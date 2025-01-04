import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, hyper_parameters):
        super(LSTMModel, self).__init__()
        self.src_vocab_size = hyper_parameters['src_vocab_size']
        self.tgt_vocab_size = hyper_parameters['tgt_vocab_size']
        self.embedding_size = hyper_parameters['lstm_hidden']
        self.hidden_size = hyper_parameters['lstm_hidden']
        self.num_layers = hyper_parameters['num_layers']

        # Embedding layers
        self.encoder_embedding = nn.Embedding(self.src_vocab_size, self.embedding_size)
        self.decoder_embedding = nn.Embedding(self.tgt_vocab_size, self.embedding_size)

        # LSTM layers
        self.encoder_lstm = nn.LSTM(
            input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(self.embedding_size, self.tgt_vocab_size)

    def forward(self, src, tgt):
        """
        src: (batch_size, seq_len)
        tgt: (batch_size, seq_len)
        """
        # Embedding
        src_embedded = self.encoder_embedding(src)  # (batch_size, seq_len, d_model)
        tgt_embedded = self.decoder_embedding(tgt)  # (batch_size, seq_len, d_model)

        # Encoder
        enc_output, (hidden, cell) = self.encoder_lstm(src_embedded)  # (batch_size, seq_len, d_model)

        # Decoder
        dec_output, _ = self.decoder_lstm(tgt_embedded, (hidden, cell))  # (batch_size, seq_len, d_model)

        # Output layer
        output = self.fc(dec_output)  # (batch_size, seq_len, tgt_vocab_size)
        output = output[:, 0, :]  # Only use the first position's output

        return output
