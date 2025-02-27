import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import dataloader, Dataset
from config.toy_config import toy_config

class sort_dataset(Dataset):
    def __init__(self, config):
        super(sort_dataset, self).__init__()
        self.config = config
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
    
    def __len__(self):
        return self.config["data"]["dataset_size"] 
    
    def __getitem__(self, idx):
        unsorted_data = torch.randint(0, 100, (self.config["model"]["input_size"],))
        sorted_data, _ = torch.sort(unsorted_data)
        return unsorted_data.to(self.device), sorted_data.to(self.device)

# usage example  
if __name__ == "__main__":
    dataset = sort_dataset(config=toy_config)
    sample = dataset[0]
    print(sample.to("cpu"))