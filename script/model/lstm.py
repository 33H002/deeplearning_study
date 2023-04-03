import torch
import torch.nn as nn
from collections import OrderedDict


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False
       
    
class LSTM(nn.Module):
    def __init__(self, *, in_features, out_features, max_seq_len,
                 num_lstm_out=100, num_lstm_layers=1):
        super().__init__()
        self.in_features = in_features
        self.max_seq_len = max_seq_len
        self.out_features = out_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(input_size=self.in_features, 
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.proj_dim = self.out_features
        projection_layers = [
            ('fc1', nn.Linear(self.num_lstm_out, self.num_lstm_out, bias=False)),
            ('bn1', nn.BatchNorm1d(self.num_lstm_out)),
            ('act', nn.GELU()),
            ('fc2', nn.Linear(self.num_lstm_out, self.proj_dim, bias=False)),
            ('bn2', BatchNorm1dNoBias(self.proj_dim)),
        ]
        self.projection = nn.Sequential(OrderedDict(projection_layers))
        
    
    def forward(self, x, seq_lens):
        x1 = nn.utils.rnn.pack_padded_sequence(x, seq_lens.cpu(), 
                                               batch_first=True, 
                                               enforce_sorted=False)
        x1, (ht,ct) = self.lstm(x1)
        x1, _ = nn.utils.rnn.pad_packed_sequence(x1, batch_first=True, 
                                                 padding_value=0.0)
        x1 = x1[:,-1,:]
        
        x_out = self.projection(x1)
        return x_out