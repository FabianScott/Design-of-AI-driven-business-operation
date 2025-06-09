import torch
import torch.nn as nn
import torch.optim as optim

class SktorchNN(nn.Module):
    def __init__(self, output_dim, hidden_layers=[64], dtype=torch.float64):
        super(SktorchNN, self).__init__()
        layers = []

        # First layer is lazy
        layers.append(nn.LazyLinear(hidden_layers[0], dtype=dtype))
        layers.append(nn.ReLU())

        # Subsequent hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i], dtype=dtype))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], output_dim, dtype=dtype))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)