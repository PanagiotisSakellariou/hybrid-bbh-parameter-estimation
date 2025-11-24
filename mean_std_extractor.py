# this script extracts the mean and std of the parameters (targets) of the training dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import os

#-------------------------------------------------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
else:
    print('No GPU found for TensorFlow.')

# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('\nUsing device:', device)

class IterTensorFlowToPyTorchDataset(IterableDataset):
    def __init__(self, tf_dataset_path):
        """
        A PyTorch dataset that fetches data from a TensorFlow dataset.
        
        Args:
        - tf_dataset: A TensorFlow dataset.
        """
        self.tf_path = tf_dataset_path
        self.tf_dataset = tf.data.Dataset.load(tf_dataset_path)
        
        
    def __len__(self):
        return tf.data.experimental.cardinality(self.tf_dataset).numpy()
            
    def __iter__(self):
        for element in self.tf_dataset.as_numpy_iterator():
            features, labels = element
            features = torch.tensor(features, dtype=torch.float32).view(1, -1)  # Reshape to match model input shape
            labels = torch.tensor(labels, dtype=torch.float32)
            
            # # Normalize the labels
            # labels = self.output_normalization(labels)
    
            yield features, labels
    
    def size(self, in_gb=False):
        ''' Returns the size of the Dataset in MB or GB.
        '''
        if os.path.exists(self.tf_path):
            if os.path.isfile(self.tf_path):
                # If it's a single file
                size_in_bytes = os.path.getsize(self.tf_path)
            elif os.path.isdir(self.tf_path):
                # If it's a directory, sum up the sizes of all files inside it
                size_in_bytes = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, _, files in os.walk(self.tf_path)
                    for file in files
                )
            else:
                raise ValueError(f"Path '{self.tf_path}' is neither a file nor a directory.")
            
                            
            if in_gb == False:
                size_in_mb = size_in_bytes / (1024*1024) # Convert to MB
                return size_in_mb
            else:
                size_in_gb = size_in_bytes / (1024*1024*1024) # Convert to GB
                return size_in_gb
        else:
            raise FileNotFoundError(f"Data file not found at {self.tf_path}")


path = '/home/sakellariou/hero_disk/test/'  # <--- Change this to the path of the dataset
#-------------------------------------------------------------------------------------------------------------------
# Load data from Tensorflow dataset

print('\nLoading data with noise')
train_dataset = IterTensorFlowToPyTorchDataset(path + 'train_dataset') 
# val_dataset = IterTensorFlowToPyTorchDataset(path + 'val_dataset')
# test_dataset = IterTensorFlowToPyTorchDataset(path + 'test_dataset')
#-------------------------------------------------------------------------------------------------------------------
  
print('\nNumber of training samples:', len(train_dataset))
# print('Number of validation samples:', len(val_dataset))
# print('Number of test samples:', len(test_dataset))

print(f'\nSize of training dataset: {train_dataset.size(in_gb=True):.3f} GB')
# print(f'Size of validation dataset: {val_dataset.size(in_gb=True):.3f} GB')
# print(f'Size of test dataset: {test_dataset.size(in_gb=True):.3f} GB')

#-------------------------------------------------------------------------------------------------------------------

# Create DataLoader for batching
print('\nCreating DataLoaders...')
batch = 1024
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False) # ItterableDataset is not shuffleable
# val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# # Check the first batch
# for inputs, targets in train_loader:
#     print('Inputs:', inputs.shape, '| Targets:', targets.shape)
#     print('Inputs:', inputs, '| Targets:', targets)
#     break

# sys.exit()

#-------------------------------------------------------------------------------------------------------------------

# extract mean and std for every target parameter from the training dataset
sum_y = 0.0
sum_sq_y = 0.0
n = 0

for _, y in train_loader:
    n += y.size(0)
    sum_y += y.sum(dim=0)
    sum_sq_y += (y ** 2).sum(dim=0)

mean = sum_y / n
std = torch.sqrt(sum_sq_y / n - mean ** 2)

print("Target mean:", mean)
print("Target std:", std)

np.save(path + 'parameters_mean.npy', mean.numpy()) # <---- Change this to the desired path
np.save(path + 'parameters_std.npy', std.numpy())   # <---- Change this to the desired path
print(f'\nMean and std saved to {path}')