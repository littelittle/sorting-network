import torch
import torch.nn as nn
from config.toy_config import toy_config



class sort_model(nn.Module):
    def __init__(self, config):
        super(sort_model, self).__init__()
        self.fc1 = nn.Linear(config['input_size'], config['hidden_size'])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config['hidden_size'], config['output_size'])
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

