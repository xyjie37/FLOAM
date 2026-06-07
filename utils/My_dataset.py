import torch  
from torch.utils.data import Dataset  
import numpy as np  
  
class AnchorDataset(Dataset):  
    def __init__(self, features, labels):
        self.len = len(features)
        self.feature = torch.from_numpy(features).float()
        self.label = torch.from_numpy(labels).long()
  
    def __len__(self):
        return self.len
  
    def __getitem__(self, idx):  
        return self.feature[idx], self.label[idx]