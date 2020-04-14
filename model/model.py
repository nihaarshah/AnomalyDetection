import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
