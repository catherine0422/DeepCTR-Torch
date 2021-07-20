import torch
import numpy as np

class NpyDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, value_file, label_file=None):
        'Initialization'
        if label_file is not None:
            self.labels = np.load(label_file)
        else:
            self.labels = None
        self.values = np.load(value_file)

  def __len__(self):
        'Denotes the total number of samples'
        return self.values.shape[0]

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.values[index,:]
        if self.labels is not None:
            y = self.labels[index, :]
            return [x, y]
        else:
            return [x]