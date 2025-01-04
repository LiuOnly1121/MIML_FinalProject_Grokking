import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        '''
        input:
            Q, K, V: matrix with shape (batch_size, num_heads, seq_length, d_k)
        output:
            a matrix with shape (batch_size, num_heads, seq_length, d_k)
        '''
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        '''
        imput:
            Q, K, L: matrix with shape (batch_size, seq_length, d_model)
        output:
            matrix with shape (batch_size, seq_length, d_model)
        '''
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    '''
    a two-layer MLP, the non-linearity is ReLU.

    does not change the shape of the Tensor
    '''
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    '''
    add the position embedding to the input

    does not change the shape of the Tensor
    '''
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    '''
    one block of encoder:

    structure:
    -> self attention
    -> adding x to the result and layer norm
    -> feed forward
    -> adding x to the result and layer norm
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    '''
    one block of decoder

    structure:
    -> self attention
    -> add x to the result and layer norm
    -> cross attention with the result of encoder
    -> add x to the result and layer norm
    -> feed forward
    -> add x to the result and layer norm
    '''
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
# class OnehotEncode(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
class Transformer(nn.Module):
    # def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0, positional_encoding=True):
    def __init__(
            self,  
            hyper_parameters = {
                'src_vocab_size': 5000, 
                'tgt_vocab_size': 5000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'd_ff': 512,
                'max_seq_length': 100,
                'dropout': 0.1,
                'positional_encoding': True
            }
            ):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(hyper_parameters['src_vocab_size'], hyper_parameters['d_model'])
        self.decoder_embedding = nn.Embedding(hyper_parameters['tgt_vocab_size'], hyper_parameters['d_model'])
        
        if hyper_parameters['positional_encoding']:
            self.positional_encoding = PositionalEncoding(hyper_parameters['d_model'], hyper_parameters['max_seq_length'])

        self.encoder_layers = nn.ModuleList([EncoderLayer(hyper_parameters['d_model'], hyper_parameters['num_heads'], hyper_parameters['d_ff'], hyper_parameters['dropout']) for _ in range(hyper_parameters['num_layers'])])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hyper_parameters['d_model'], hyper_parameters['num_heads'], hyper_parameters['d_ff'], hyper_parameters['dropout']) for _ in range(hyper_parameters['num_layers'])])

        self.fc = nn.Linear(hyper_parameters['d_model'], hyper_parameters['tgt_vocab_size'])
        self.dropout = nn.Dropout(hyper_parameters['dropout'])

    def generate_mask(self, src, tgt):
        device = src.device
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        if self.positional_encoding:
            src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        else:
            src_embedded = self.dropout(self.encoder_embedding(src))
            tgt_embedded = self.dropout(self.decoder_embedding(tgt))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        output = output[:,0,:]
        return output