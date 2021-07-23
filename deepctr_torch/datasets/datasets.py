import torch
import numpy as np

class NpyDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, value_file, label_file=None, p_sample=1):
        'Initialization'
        if p_sample > 1 or p_sample <= 0:
            raise ValueError('Portion of sample of dataset should be within (0,1], current p_sample = ', p_sample)
        if label_file is not None:
            self.labels = np.load(label_file)
        else:
            self.labels = None
        self.values = np.load(value_file)
        if p_sample < 1:
            n_total = self.values.shape[0]
            n_sample = round(n_total * p_sample)
            self.values = self.values[:n_sample, :]
            self.labels = self.labels[:n_sample, :] if self.labels is not None else None

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