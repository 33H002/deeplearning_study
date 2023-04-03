import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features=331, hidden_dim=128, out_features=1, num_layers=4, dropout=0.):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = nn.GELU()

        self.layers = nn.ModuleList([nn.Linear(self.in_features, self.hidden_dim), self.activation]) 
        for i in range(self.num_layers - 1):
            self.layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), 
                                nn.LayerNorm(self.hidden_dim), 
                                self.activation, 
                                nn.Dropout(self.dropout)])
        self.layers.append(nn.Linear(self.hidden_dim, self.out_features)) 
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x