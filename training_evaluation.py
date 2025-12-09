# This script trains and evaluates a model. Requires GPU access for PyTorch.
# see the <---- comments before you run
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import os
import math
from torchinfo import summary


# Limit TensorFlow GPU memory usage
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

# Check if GPU is available for pytorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('\nUsing device:', device)

# This class loads the tensorflow dataset and turns each element into torch tensors
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


path = '/home/sakellariou/hero_disk/test/'  # <---- Change this to the path of the datasets

#-------------------------------------------------------------------------------------------------------------------
# Load data from Tensorflow dataset
print('\nLoading data with noise')
train_dataset = IterTensorFlowToPyTorchDataset(path + 'train_dataset') # <---- Change the name of the datasets if needed
val_dataset = IterTensorFlowToPyTorchDataset(path + 'val_dataset')     # <----
test_dataset = IterTensorFlowToPyTorchDataset(path + 'test_dataset')   # <----
  
print('\nNumber of training samples:', len(train_dataset))
print('Number of validation samples:', len(val_dataset))
print('Number of test samples:', len(test_dataset))

print(f'\nSize of training dataset: {train_dataset.size(in_gb=True):.3f} GB')
print(f'Size of validation dataset: {val_dataset.size(in_gb=True):.3f} GB')
print(f'Size of test dataset: {test_dataset.size(in_gb=True):.3f} GB')
#-------------------------------------------------------------------------------------------------------------------
# Create DataLoader for batching
print('\nCreating DataLoaders...')
batch = 1024        # <---- Change the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False) # ItterableDataset is not shuffleable
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# # Check the first batch
# for inputs, targets in train_loader:
#     print('Inputs:', inputs.shape, '| Targets:', targets.shape)
#     print('Inputs:', inputs, '| Targets:', targets)
#     break

# sys.exit()

# -----------------------------------------------------------------------------------------------------------------
# Define the models -----------------------------------------------------------------------------------------------

# ShallowModel
class ShallowModel(nn.Module):
    """
    Shallow CNN model for regression.
    
    The model requires initialization before loading weights. 
    To initialize the model, call model.initialize(input_tensor).
    """
    def __init__(self, num_vars):
        super(ShallowModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=1, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=1, dilation=4)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=8, stride=1, dilation=4)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.flatten = nn.Flatten()
        self.fc1 = None # This layer will be initialized in the forward method
        self.fc2 = nn.Linear(64, num_vars)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        
        # Initialize the first fully connected layer the first time forward is run
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 64).to(x.device) 
            
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("\nModel initialized successfully.")

# DeepModel
class DeepModel(nn.Module):
    """
    Deep CNN model for regression.
    
    The model requires initialization before loading weights. 
    To initialize the model, call model.initialize(input_tensor).
    """
    def __init__(self, num_vars):
        super(DeepModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=16, stride=1, dilation=1)
        # self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=16, stride=1, dilation=2)
        # self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=16, stride=1, dilation=2)
        # self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=32, stride=1, dilation=2)
        # self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.flatten = nn.Flatten()
        self.fc1 = None # This layer will be initialized in the forward method
        # self.drp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        # self.drp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_vars)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        # x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        # x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)
        
        x = self.flatten(x)
        
        # Initialize the first fully connected layer the first time forward is run
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
            
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.drp1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.drp2(x)
        
        x = self.fc3(x)
        return x

    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("\nModel initialized successfully.")

# BNSModel
class BNSModel(nn.Module):
    """
    BNS CNN model for regression based on the paper
    ~Detection and parameter estimation of gravitational 
    waves from binary neutron-star mergers in real LIGO 
    data using deep learning~.
    
    The model requires initialization before loading weights. 
    To initialize the model, call model.initialize(input_tensor).
    """
    def __init__(self, num_vars):
        super(BNSModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=16)
        # self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8)
        # self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8)
        # self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=8)
        # self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        
        self.flatten = nn.Flatten()
        self.fc1 = None # This layer will be initialized in the forward method
        # self.drp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        # self.drp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_vars)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        # x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        # x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)
        
        x = self.flatten(x)
        
        # Initialize the first fully connected layer the first time forward is run
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)
            
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.drp1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.drp2(x)
        
        x = self.fc3(x)
        return x

    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("\nModel initialized successfully.")

# ShallowModel-hybrid
class HybridShallowModel(nn.Module):
    def __init__(self, num_vars):
        super(HybridShallowModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=16, stride=1, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=1, dilation=4)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=8, stride=1, dilation=4)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4) # embedding size = 64, 4 heads
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_vars)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        # Prepare for Transformer
        # Transpose to match Encoder input shape: (sequence_length, batch_size, embed_dim)
        x = x.permute(2, 0, 1)
        
        x = self.transformer(x)  # Output: (sequence_length, batch_size, embed_dim)
        x = x[-1, :, :]  # Take the last sequence element (batch_size, embed_dim) 
            
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("\nModel initialized successfully.")

# BNSModel-hybrid
class HybridBNSModel(nn.Module):
    def __init__(self, num_vars):
        super(HybridBNSModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=16)
        # self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8)
        # self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=8)
        # self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=8)
        # self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=4)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4) # embedding size = 128, 4 heads
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        
        self.fc1 = nn.Linear(256, 128)
        # self.drp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        # self.drp2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_vars)
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        # x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        # x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)
        
        # Prepare for Transformer
        # Transpose to match Encoder input shape: (sequence_length, batch_size, embed_dim)
        x = x.permute(2, 0, 1)
        
        x = self.transformer(x)  # Output: (sequence_length, batch_size, embed_dim)
        x = x[-1, :, :]  # Take the last sequence element (batch_size, embed_dim)
            
        x = self.fc1(x)
        x = torch.relu(x)
        # x = self.drp1(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        # x = self.drp2(x)
        
        x = self.fc3(x)
        return x

    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("\nModel initialized successfully.")
            
# DeepModel-hybrid
class HybridDeepModel(nn.Module):
    def __init__(self, num_variables):
        super(HybridDeepModel, self).__init__()
        
        # CNN Layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=16, stride=1, dilation=1)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=16, stride=1, dilation=2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=16, stride=1, dilation=2)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=32, stride=1, dilation=2)
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)

        # Normalization before Transformer
        self.norm = nn.LayerNorm(512)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 128)  # Adjust input size based on the output of the last conv layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_variables)
    
    def forward(self, x):
        # CNN Feature Extraction
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = torch.relu(x)   
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.pool4(x) 
        
        # Prepare for Transformer
        # Transpose to match Encoder input shape: (batch_size, sequence_length, embed_dim)
        x = x.permute(0, 2, 1)

        x = self.norm(x)

        x = self.transformer(x)  # Output: (sequence_length, batch_size, embed_dim)
        x = x[:, -1, :]  # Take the last sequence element (batch_size, embed_dim)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)

        return x
    
    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("Model initialized successfully.")

        
#-------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------
# Initialize model, loss function, and optimizer
num_vars = 6  # The model will predict six parameters
model = HybridDeepModel(num_vars).to(device) # <---- Change to one of the models above
criterion = nn.L1Loss()                      # <---- Change Loss Function here. L1Loss is MAE
optimizer = optim.Adam(model.parameters(), lr=0.0001) # <---- Change Learning rate as needed.

## Create the scheduler, monitoring validation loss
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-5)

early_stop = False         # Set to True to enable early stopping or False to disable
early_stop_patience = 50

# Print model's name
print(f'\nModel : {model.__class__.__name__}')
print(f'\nOptimizer: {optimizer.__class__.__name__} || Initial learning rate: {optimizer.param_groups[0]["lr"]}')

# Remove the comment below to view layer shapes and parameters of the model.
# summary(model, input_size=(1, 1, 8192))  # (batch_size, channels, length)
# sys.exit()

# Create the destination folder if it doesn't exist
destination_folder = f'./test/{model.__class__.__name__}_results'  # <---- Change the result folder here.
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
# Training function
def train_model(model, train_loader, val_loader, epochs, resume_checkpoint=None):
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    early_stop_count = 0
    start_epoch = 1 # Start from epoch 1 by default
    
    prev_lr = optimizer.param_groups[0]['lr']

    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging
    
    # Load checkpoint if resuming
    if resume_checkpoint is not None:
        try:
            checkpoint = torch.load(resume_checkpoint)
            model_loaded = False
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                model_loaded = True
                print('\nModel weights loaded successfully.')
              
            except Exception as e:
                print(f'\nError loading model weights : {e}')
                print('Attempting to initialize model...')
                with torch.no_grad():
                    inputs, targets = next(iter(train_loader))
                    inputs = inputs.to(device)
                    model.initialize(inputs)
                model.load_state_dict(checkpoint['model_state_dict'])
                model_loaded = True
                print('\nModel weights loaded after manual initialization.')
                
            except RuntimeError as e:
                print(f'\n Model architecture mismatch: {e}')
                print('Skipping checkpoint. Training will start from scratch.')
            
            if model_loaded:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])   # dont forget to comment this line if no scheduler is used
                prev_lr = optimizer.param_groups[0]['lr']
                start_epoch = checkpoint['epoch'] + 1
                train_losses = checkpoint['train_losses']
                val_losses = checkpoint['val_losses']
                best_val_loss = min(val_losses) if val_losses else np.inf
                best_val_epoch = val_losses.index(best_val_loss) + 1 if val_losses else 0
                best_train_loss = min(train_losses) if train_losses else np.inf
                best_train_epoch = train_losses.index(best_train_loss) + 1 if train_losses else 0
                
                print(f'\nResuming training from epoch {start_epoch}.')
                print(f'Learning Rate: {prev_lr}')
                print(f'Best Validation Loss: {best_val_loss} on epoch {best_val_epoch}')
                print(f'Best Training Loss: {best_train_loss} on epoch {best_train_epoch}')
                
        except Exception as e:
            print(f'\nError loading checkpoint: {e}')
            print('Training will start from scratch.')
            
    print('\nStarting Training phase...')
    
    for epoch in range(start_epoch, epochs+1):
        model.train()
        running_loss = 0.0
        counter = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", dynamic_ncols=True)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Check for NaNs in input or target
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                raise ValueError("NaN detected in inputs or targets.")
            
            optimizer.zero_grad()
            outputs = model(inputs)

            # Check for NaNs in outputs
            if torch.isnan(outputs).any():
                raise ValueError("NaN detected in model outputs.")
            
            loss = criterion(outputs, targets)
            
            # Check if loss is NaN or Inf
            if torch.isnan(loss) or math.isinf(loss.item()):
                print("Loss is NaN or Inf!")
                print("Inputs stats:", inputs.min().item(), inputs.max().item())
                print("Outputs stats:", outputs.min().item(), outputs.max().item())
                print("Targets stats:", targets.min().item(), targets.max().item())
                continue
            
            loss.backward()
            
            # Gradient check
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradient of {name}")
                    raise ValueError("NaN detected in gradients.")
                
            optimizer.step()
            
            running_loss += loss.item()*inputs.size(0)
            counter += inputs.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({'Train Loss': f'{running_loss / counter:.4f}'})
            
            
        train_loss = running_loss / counter
        train_losses.append(train_loss)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        val_counter = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                val_counter += inputs.size(0)

        val_loss /= val_counter
        val_losses.append(val_loss)
                
        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            progress_bar.write(f'Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...')
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{destination_folder}/{model.__class__.__name__}_best_model.pth')
            early_stop_count = 0
        else:
            progress_bar.write(f'Epoch {epoch}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Validation loss did not improve from {best_val_loss:.4f}')
            early_stop_count += 1
        
        # # Step the scheduler with the validation loss
        # scheduler.step(val_loss)  # Reduces learning rate if validation loss plateaus
        
        # # Get the current learning rate
        # current_lr = optimizer.param_groups[0]['lr']
        
        # # Print only if learning rate has changed
        # if current_lr != prev_lr:
        #     print(f"Learning Rate changed: {prev_lr} -> {current_lr}")
        #     prev_lr = current_lr  # Update previous learning rate
          
        # Save a checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),  # comment this if there is no scheduler
            'train_losses': train_losses,
            'val_losses': val_losses
            }, f'{destination_folder}/{model.__class__.__name__}_checkpoint.pth')
        
        # Check for early stopping 
        if early_stop == True and early_stop_count >= early_stop_patience:
            progress_bar.write(f'Early stopping triggered after {early_stop_patience} epochs without the validation loss improving.')
            break
        
    print('\nTraining complete!')
    print(f'\nModel: {model.__class__.__name__}')
    print(f'\nBest Training Loss: {min(train_losses)} in epoch {train_losses.index(min(train_losses)) + 1}')
    print(f'Best Validation Loss: {min(val_losses)} in epoch {val_losses.index(min(val_losses)) + 1}')
        
    return train_losses, val_losses

# Train the model
epochs = 150  # <---- Change the number of Epochs here.
checkpoint = None #f'{destination_folder}/{model.__class__.__name__}_checkpoint.pth' # uncomment this if you want to continue training from checkpoint.
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, resume_checkpoint=checkpoint)

# <--- Remove the comment below to generate plots up to the checkpoint. (If the train_model function terminates due to an interruption.)
# # Load model from checkpoint
# checkpoint = torch.load(f'{destination_folder}/{model.__class__.__name__}_checkpoint.pth')
# try:
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print('\nModel weights loaded successfully.')
# except:
#     print('\nError loading model weights. Initializing model...')
#     with torch.no_grad():
#         inputs, targets = next(iter(train_loader))
#         inputs = inputs.to(device)
#         model.initialize(inputs)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     print('\nModel weights loaded successfully.')
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# train_losses = checkpoint['train_losses']
# val_losses = checkpoint['val_losses']

# Plot the training and validation losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'{model.__class__.__name__} Loss')
plt.grid()
plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_loss.png', dpi=300, bbox_inches='tight')

# Load the best model weights
print('\nLoading best model weights...')
try:
    model.load_state_dict(torch.load(f'{destination_folder}/{model.__class__.__name__}_best_model.pth'))
    print('\nModel weights loaded successfully.')
except:
    print('\nError loading model weights. Initializing model...')
    with torch.no_grad():
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(device)
        model.initialize(inputs)
    model.load_state_dict(torch.load(f'{destination_folder}/{model.__class__.__name__}_best_model.pth'))
    print('\nModel weights loaded successfully.')

# Test the model -----------------------------------------------------------------------------------------------
def evaluate_model(model, test_loader, num_vars):
    print('\nMaking evaluation metrics...')

    # Set model to evaluation mode
    model.eval()
    
    # # Get the dataset instance from the dataloader
    # dataset_instance = test_loader.dataset
    
    # Initialize metric accumulators
    mae_sum = np.zeros(num_vars)
    mse_sum = np.zeros(num_vars)
    mare_sum = np.zeros(num_vars)
    mre_sum = np.zeros(num_vars)
    total_y_true = np.zeros(num_vars)
    # total_y_pred = np.zeros(num_vars)
    total_y_true_squared = np.zeros(num_vars)
    count = 0
    
    with torch.no_grad():
        
        progress_bar = tqdm(test_loader, dynamic_ncols=True)
        
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Make predictions
            test_predictions = model(inputs)
            
            # Convert to numpy arrays
            y_true = targets.cpu().numpy()
            y_pred = test_predictions.cpu().numpy()

            # Update accumulators
            e = 1e-8 # Small value to prevent division by zero
                        
            mae_sum += np.sum(np.abs(y_true - y_pred), axis=0)
            mse_sum += np.sum((y_true - y_pred) ** 2, axis=0)
            mare_sum += np.sum(np.abs((y_true - y_pred) / (y_true + e)), axis=0)
            mre_sum += np.sum((y_true - y_pred) / (y_true + e), axis=0)
            total_y_true += np.sum(y_true, axis=0)
            # total_y_pred += np.sum(y_pred, axis=0)
            total_y_true_squared += np.sum(y_true** 2, axis=0)
            # total_errors += np.sum(np.abs(y_true - y_pred), axis=0)
            count += y_true.shape[0]
            
                        
    # Calculate final metrics
    mae = mae_sum / count
    mse = mse_sum / count
    rmse = np.sqrt(mse)
    mare = mare_sum / count
    mape = mare*100
    mre = mre_sum / count
    mpe = mre * 100
    
    r2 = 1 - (mse_sum / (total_y_true_squared - (total_y_true ** 2 / count)))
    
    total_mae_loss = np.sum(mae)/num_vars
    
    # Get model name from class
    model_name = model.__class__.__name__
    
    print(f'Model: {model_name}')
    print(f'Test Loss: {total_mae_loss}')

    # Metrics dictionary
    metrics = {
        'model_name': model_name,
        'loss': total_mae_loss.tolist(),
        'mae': mae.tolist(),
        'mape': mape.tolist(),
        'mare': mare.tolist(),
        'mre': mre.tolist(),
        'mpe': mpe.tolist(),        # If MPE is negative, the model tends to predict values higher than actual.
        'rmse': rmse.tolist(),
        'mse': mse.tolist(),
        'r2': r2.tolist(),
    }


    # Save metrics as JSON
    with open(f'{destination_folder}/{model.__class__.__name__}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

# Make evaluation metrics
metrics = evaluate_model(model, test_loader, num_vars)

def print_metrics(metrics):
    # Print the evaluation metrics
    print()
    print('-' * 50)
    print("Mean Absolute Error (MAE):")
    print(f"\tMass 1: {metrics['mae'][0]}")
    print(f"\tMass 2: {metrics['mae'][1]}")
    print(f"\tDistance: {metrics['mae'][2]}")
    print(f'\tInclination: {metrics["mae"][3]}')
    print(f'\tSpin 1: {metrics["mae"][4]}')
    print(f'\tSpin 2: {metrics["mae"][5]}')

    print()
    print('-' * 50)
    print("Mean Absolute Percentage Error (MAPE):")
    print(f"\tMass 1: {metrics['mape'][0]}")
    print(f"\tMass 2: {metrics['mape'][1]}")
    print(f"\tDistance: {metrics['mape'][2]}")
    print(f'\tInclination: {metrics["mape"][3]}')
    print(f'\tSpin 1: {metrics["mape"][4]}')
    print(f'\tSpin 2: {metrics["mape"][5]}')

    print()
    print('-' * 50)
    print("Mean Relative Error (MRE):")
    print(f"\tMass 1: {metrics['mre'][0]}")
    print(f"\tMass 2: {metrics['mre'][1]}")
    print(f"\tDistance: {metrics['mre'][2]}")
    print(f'\tInclination: {metrics["mre"][3]}')
    print(f'\tSpin 1: {metrics["mre"][4]}')
    print(f'\tSpin 2: {metrics["mre"][5]}')

    print()
    print('-' * 50)
    print("Mean Percentage Error (MPE):")
    print(f"\tMass 1: {metrics['mpe'][0]}")
    print(f"\tMass 2: {metrics['mpe'][1]}")
    print(f"\tDistance: {metrics['mpe'][2]}")
    print(f'\tInclination: {metrics["mpe"][3]}')
    print(f'\tSpin 1: {metrics["mpe"][4]}')
    print(f'\tSpin 2: {metrics["mpe"][5]}')

    print()
    print('-' * 50)
    print("Root Mean Squared Error (RMSE):")
    print(f"\tMass 1: {metrics['rmse'][0]}")
    print(f"\tMass 2: {metrics['rmse'][1]}")
    print(f"\tDistance: {metrics['rmse'][2]}")
    print(f'\tInclination: {metrics["rmse"][3]}')
    print(f'\tSpin 1: {metrics["rmse"][4]}')
    print(f'\tSpin 2: {metrics["rmse"][5]}')

    print()
    print('-' * 50)
    print("Mean Squared Error (MSE):")
    print(f"\tMass 1: {metrics['mse'][0]}")
    print(f"\tMass 2: {metrics['mse'][1]}")
    print(f"\tDistance: {metrics['mse'][2]}")
    print(f'\tInclination: {metrics["mse"][3]}')
    print(f'\tSpin 1: {metrics["mse"][4]}')
    print(f'\tSpin 2: {metrics["mse"][5]}')

    print()
    print('-' * 50)
    print("R^2 Score:")
    print(f"\tMass 1: {metrics['r2'][0]}")
    print(f"\tMass 2: {metrics['r2'][1]}")
    print(f"\tDistance: {metrics['r2'][2]}")
    print(f'\tInclination: {metrics["r2"][3]}')
    print(f'\tSpin 1: {metrics["r2"][4]}')
    print(f'\tSpin 2: {metrics["r2"][5]}')
    print('-' * 50)
    print()


def prediction_viewer(model, dataloader, start_line, end_line):
    """
    Visualizes model predictions on a dataset in PyTorch.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: A PyTorch DataLoader containing test data.
        start_line: Starting index for lines to print.
        end_line: Ending index for lines to print.
    """
    global_index = 0  # Tracks the global index of each sample
    printed_count = 0  # Tracks how many specified lines have been printed

    # # Get the dataset instance from the dataloader
    # dataset_instance = dataloader.dataset
    
    # Set the model to evaluation mode
    model.eval()
    
    # Ensure lines_to_print is a set for efficient membership testing
    lines_to_print = set(range(start_line, end_line + 1))
    
    print('\nPrinting predictions...')
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for x_batch, y_batch in dataloader:
            # Move to appropriate device if necessary
            x_batch = x_batch.to(next(model.parameters()).device)
            y_batch = y_batch.to(next(model.parameters()).device)
            
            # Make predictions on the current batch
            test_predictions = model(x_batch)

            # Convert to numpy arrays
            y_true_batch = y_batch.cpu().numpy()
            y_pred_batch = test_predictions.cpu().numpy()

            for i in range(y_true_batch.shape[0]):
                if global_index in lines_to_print:
                    print(f"Line {global_index}: True values: {y_true_batch[i]} || Predicted values: {y_pred_batch[i]}")
                    printed_count += 1

                    # Stop if all specified lines have been printed
                    if printed_count == len(lines_to_print):
                        return

                global_index += 1


# Print metrics
print_metrics(metrics)
               
# Specify the lines you want to print
start_line = 0
end_line = 20

# Call the prediction_viewer function
prediction_viewer(model, test_loader, start_line, end_line)
