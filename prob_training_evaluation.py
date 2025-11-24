# here i train the model on negative log likelihood (freeze encoder approach)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
from nflows.distributions import StandardNormal
from nflows import transforms, distributions, flows
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import os
from collections import defaultdict
import pandas as pd
import math
import properscoring as ps
from scipy.stats import kstest, combine_pvalues

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
        self.mean = torch.tensor(np.load(path + 'parameters_mean.npy'), dtype=torch.float32) # <---- change path if needed
        self.std = torch.tensor(np.load(path + 'parameters_std.npy'), dtype=torch.float32)   # <---- change path if needed


    def __len__(self):
        return tf.data.experimental.cardinality(self.tf_dataset).numpy()
            
    def __iter__(self):
        for element in self.tf_dataset.as_numpy_iterator():
            features, labels = element
            features = torch.tensor(features, dtype=torch.float32).view(1, -1)  # Reshape to match model input shape
            labels = torch.tensor(labels, dtype=torch.float32)
            
            # Normalize the labels    
            labels = (labels - self.mean) / self.std
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
val_dataset = IterTensorFlowToPyTorchDataset(path + 'val_dataset')
test_dataset = IterTensorFlowToPyTorchDataset(path + 'test_dataset')
  
print('\nNumber of training samples:', len(train_dataset))
print('Number of validation samples:', len(val_dataset))
print('Number of test samples:', len(test_dataset))

print(f'\nSize of training dataset: {train_dataset.size(in_gb=True):.3f} GB')
print(f'Size of validation dataset: {val_dataset.size(in_gb=True):.3f} GB')
print(f'Size of test dataset: {test_dataset.size(in_gb=True):.3f} GB')
#-------------------------------------------------------------------------------------------------------------------

# Create DataLoader for batching
print('\nCreating DataLoaders...')
batch = 1024
print(f'Batch size: {batch}')
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False) # ItterableDataset is not shuffleable
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
print('DataLoaders created successfully!')

# Check the first batch
for inputs, targets in train_loader:
    print('Inputs:', inputs.shape, '| Targets:', targets.shape)
    print('Inputs:', inputs, '| Targets:', targets)
    break

sys.exit()

# Define the models -----------------------------------------------------------------------------------------------

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
        x = self._latent_space_extractor(x)

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
    
    def _latent_space_extractor(self, x):
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
        return x
    
    def encode(self, x):
        feature = self._latent_space_extractor(x)
        # Initialize the first fully connected layer the first time forward is run
        if self.fc1 is None:
            self.fc1 = nn.Linear(feature.size(1), 128).to(feature.device)

        c = self.fc1(feature)
        c = torch.relu(c)
        # c = self.drp1(c)

        c = self.fc2(c)
        c = torch.relu(c)
        # c = self.drp2(c)

        return c

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
        features = self._latent_space_extractor(x)
        
        # Fully Connected Layers
        x = self.fc1(features)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)

        return x
    
    def _latent_space_extractor(self, x):
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
        
        return x
    
    def encode(self, x):
        # returns context vector for the flow
        feature = self._latent_space_extractor(x)
        c = self.fc1(feature)
        c = torch.relu(c)
        
        c = self.fc2(c)
        c = torch.relu(c)
        
        return c

    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.forward(x)
        print("Model initialized successfully.")


class DeepFlowModelv3(nn.Module):
    def __init__(self, num_variables, freeze_encoder=False):
        super(DeepFlowModelv3, self).__init__()
        self.encoder = DeepModel(num_variables).to(device)  # Adjust num_variables as needed

        if freeze_encoder:
            print("\nFreezing the encoder weights.")
            # <---- change path if needed
            encoder_weights_path = f'./{self.encoder.__class__.__name__}_results/{self.encoder.__class__.__name__}_best_model.pth'
            if os.path.exists(encoder_weights_path):
                with torch.no_grad():
                    inputs, _ = next(iter(train_loader))
                    inputs = inputs.to(device)
                    self.encoder.initialize(inputs)
                self.encoder.load_state_dict(torch.load(encoder_weights_path))
                print(f"\nEncoder weights loaded from {encoder_weights_path}")
            else:
                print(f"\nEncoder weights not found at {encoder_weights_path}")

            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        transform_list = []
        for _ in range(num_variables):
            transform_list.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=num_variables, 
                    hidden_features=128, 
                    context_features=64
                )
            )
            transform_list.append(transforms.RandomPermutation(features=num_variables))
        
        transform = transforms.CompositeTransform(transform_list)
        base_distribution = distributions.StandardNormal(shape=[num_variables])
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

    def forward(self, waveform, parameters=None, num_samples=1, return_point_estimate=False):
        # Extract context vector from the encoder
        encoded = self.encoder.encode(waveform) # shape (batch_size, context_dim) 

        if return_point_estimate:
            # Predict point estimate using the encoder + FC layers from DeepModel
            point_estimate = self.encoder.forward(waveform)
            return point_estimate
        
        if parameters is not None:
            # Training mode: return log prob of given parameters under posterior
            # parameters shape: (batch_size, num_variables)
            log_prob = self.flow.log_prob(inputs=parameters, context=encoded)
            return log_prob
        else:
            # Sampling mode: generate samples from posterior conditioned on context
            samples = self.flow.sample(num_samples=num_samples, context=encoded)
            # samples shape: (num_samples, batch_size, num_variables)
            return samples
    
    def initialize(self, x):
        """Initialize the model by passing an input tensor."""
        self.encoder.forward(x)
        print("\nModel initialized successfully.")

class HybridDeepFlowModelv3(nn.Module):
    def __init__(self, num_variables, freeze_encoder=False):
        super(HybridDeepFlowModelv3, self).__init__()
        self.encoder = HybridDeepModel(num_variables)  # Adjust num_variables as needed
        
        if freeze_encoder:
            print("\nFreezing the encoder weights.")
            # <----- change path if needed
            encoder_weights_path = f'./{self.encoder.__class__.__name__}_results/{self.encoder.__class__.__name__}_best_model.pth'
            if os.path.exists(encoder_weights_path):
                self.encoder.load_state_dict(torch.load(encoder_weights_path))
                print(f"\nEncoder weights loaded from {encoder_weights_path}")
            else:
                print(f"\nEncoder weights not found at {encoder_weights_path}")

            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        transform_list = []
        for _ in range(num_variables):
            transform_list.append(
                transforms.MaskedAffineAutoregressiveTransform(
                    features=num_variables, 
                    hidden_features=128, 
                    context_features=64
                )
            )
            transform_list.append(transforms.RandomPermutation(features=num_variables))
        
        transform = transforms.CompositeTransform(transform_list)
        base_distribution = distributions.StandardNormal(shape=[num_variables])
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

    def forward(self, waveform, parameters=None, num_samples=1, return_point_estimate=False):
        # Extract context vector from the encoder
        encoded = self.encoder.encode(waveform) # shape (batch_size, context_dim) 

        if return_point_estimate:
            # Predict point estimate using the encoder + FC layers from HybridDeepModel
            point_estimate = self.encoder.forward(waveform)
            return point_estimate
        
        if parameters is not None:
            # Training mode: return log prob of given parameters under posterior
            # parameters shape: (batch_size, num_variables)
            log_prob = self.flow.log_prob(inputs=parameters, context=encoded)
            return log_prob
        else:
            # Sampling mode: generate samples from posterior conditioned on context
            samples = self.flow.sample(num_samples=num_samples, context=encoded)
            # samples shape: (num_samples, batch_size, num_variables)
            return samples
        
        
#-------------------------------------------------------------------------------------------------------------------
# Initialize model, loss function, and optimizer
num_vars = 6
model = DeepFlowModelv3(num_vars, freeze_encoder=True).to(device)
# criterion = nn.L1Loss() 
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) #lr=1e-4 for DeepModel, lr=1e-6 for DeepModel-hybrid

## Create the scheduler, monitoring validation loss
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-5)

early_stop = False         # Set to True to enable early stopping or False to disable
early_stop_patience = 50

# Print model's name
print(f'\nModel : {model.__class__.__name__}')
print(f'\nOptimizer: {optimizer.__class__.__name__} || Initial learning rate: {optimizer.param_groups[0]["lr"]}')

# Create the destination folder if it doesn't exist
destination_folder = f'./{model.__class__.__name__}_results/' # <--- change path if needed
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
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print("NaN in", name)
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                for n, v in optimizer.state.items():
                    for k, t in v.items():
                        if torch.is_tensor(t) and torch.isnan(t).any():
                            print("NaN in optimizer state:", n, k)
                            
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
            
            log_prob = model(inputs, parameters=targets) #output

            # Check for NaNs in outputs
            if torch.isnan(log_prob).any():
                raise ValueError("NaN detected in model outputs.")
            
            loss = -log_prob.mean()  # Negative log likelihood loss
            
            # Check if loss is NaN or Inf
            if torch.isnan(loss) or math.isinf(loss.item()):
                print("Loss is NaN or Inf!")
                print("Inputs stats:", inputs.min().item(), inputs.max().item())
                print("Outputs stats:", log_prob.min().item(), log_prob.max().item())
                print("Targets stats:", targets.min().item(), targets.max().item())
                continue
            
            loss.backward()
            
            # Gradient check
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradient of {name}")
                    raise ValueError("NaN detected in gradients.")
                
            optimizer.step()
            
            running_loss += loss.item()*inputs.size(0)         # Accumulate loss for the epoch
            counter += inputs.size(0)                          # Increment counter by the batch size

            # Update progress bar
            progress_bar.set_postfix({'Train Loss': f'{running_loss/counter:.4f}'})


        train_loss = running_loss / counter
        train_losses.append(train_loss)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        val_counter = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                log_prob = model(inputs, parameters=targets)
                loss = -log_prob.mean()
                val_loss += loss.item() * inputs.size(0)

                val_counter += inputs.size(0)

        val_loss /= val_counter
        
        if val_loss < 0 :
            print(f'Validation loss is negative: {val_loss:.4f}')
            print('The training process will be stopped.')
            break
        
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
epochs = 150
checkpoint = None #f'{destination_folder}/{model.__class__.__name__}_checkpoint.pth'
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs, resume_checkpoint=checkpoint)

# # Load model from checkpoint
# print('\nLoading model from checkpoint...')
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


def evaluate_model(model, test_loader, num_vars):
    print('\nMaking evaluation metrics...')
    model.eval()
    
    mean = np.load(path + 'parameters_mean.npy') # <---- Change path if needed
    std = np.load(path + 'parameters_std.npy')   # <---- Change path if needed
    
    # Accumulators for point estimates
    mae_sum = np.zeros(num_vars)
    mse_sum = np.zeros(num_vars)
    mare_sum = np.zeros(num_vars)
    mre_sum = np.zeros(num_vars)
    total_y_true = np.zeros(num_vars)
    total_y_true_squared = np.zeros(num_vars)
    count = 0
    
    # For probabilistic metric
    total_nll = 0.0

    # Per-Mass_1 accumulators
    per_mass_1_mae = defaultdict(lambda: np.zeros(num_vars))
    per_mass_1_sq = defaultdict(lambda: np.zeros(num_vars))
    per_mass_1_count = defaultdict(int)
    
    # Per-Mass_2 accumulators
    per_mass_2_mae = defaultdict(lambda: np.zeros(num_vars))
    per_mass_2_sq = defaultdict(lambda: np.zeros(num_vars))
    per_mass_2_count = defaultdict(int)
    
    # Per-distance accumulators
    per_distance_mae = defaultdict(lambda: np.zeros(num_vars))
    per_distance_sq = defaultdict(lambda: np.zeros(num_vars))
    per_distance_count = defaultdict(int)

    # Per-inclination accumulators
    per_inclination_mae = defaultdict(lambda: np.zeros(num_vars))
    per_inclination_sq = defaultdict(lambda: np.zeros(num_vars))
    per_inclination_count = defaultdict(int)
    
    # Per-spin_1 accumulators
    per_spin_1_mae = defaultdict(lambda: np.zeros(num_vars))
    per_spin_1_sq = defaultdict(lambda: np.zeros(num_vars))
    per_spin_1_count = defaultdict(int)

    # Per-spin_2 accumulators
    per_spin_2_mae = defaultdict(lambda: np.zeros(num_vars))
    per_spin_2_sq = defaultdict(lambda: np.zeros(num_vars))
    per_spin_2_count = defaultdict(int)
    
    # Per-value accumulators
    mass_1_index = 0  # Mass 1 is the first variable (index 0)
    mass_2_index = 1  # Mass 2 is the second variable (index 1)
    distance_index = 2  # Distance is the third variable (index 2)
    inclination_index = 3  # Inclination is the fourth variable (index 3)
    spin_1_index = 4  # Spin 1 is the fifth variable (index 4)
    spin_2_index = 5  # Spin 2 is the sixth variable (index 5)

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Compute log_prob of true parameters for NLL
            log_prob = model(inputs, parameters=targets)
            nll = -log_prob.mean().item()
            total_nll += nll * inputs.size(0)

            # Sample from flow posterior to get point estimates (mean)
            samples = []
            samples_per_input = 100   # 1000 is better for P-P plots but slow
            samples_per_batch = 10
            num_batches = samples_per_input // samples_per_batch
            
            for _ in range(num_batches):  # batch sampling for speed
                s = model(inputs, parameters=None, num_samples=samples_per_batch) # (batch, samples_per_batch, num_vars)
                samples.append(s)
                
            samples = torch.cat(samples, dim=1) # (batch, num_batches*samples_per_batch, num_vars)
            
            samples = samples.cpu().numpy()  # Convert to numpy for further processing
            posterior_means = samples.mean(axis=1)
            y_true = targets.cpu().numpy()

            # Denormalize
            y_true_denorm = y_true*std + mean
            posterior_means_denorm = posterior_means*std + mean
            samples_denorm = samples*std + mean
            
            # Compute CRPS per parameter, per sample
            
            crps_scores = [
                ps.crps_ensemble(y_true_denorm[:, var_idx], samples_denorm[:, :, var_idx]).mean()
                for var_idx in range(samples_denorm.shape[2])
            ]

            # Sharpness (average posterior std)
            sharpness = np.std(samples_denorm, axis=1).mean(axis=0)
            
            # Probability Integral Transform (PIT)
            pit_values_per_param = [
                (samples_denorm[:, :, var_idx] <= y_true_denorm[:, var_idx, None]).mean(axis=1)
                for var_idx in range(num_vars)
            ]
            
            pit_values = np.concatenate(pit_values_per_param)        
            global_PIT_bias = np.abs(np.mean(pit_values) - 0.5)

            credible_levels = np.array([0.5, 0.7, 0.9, 0.95])
            calibration_errors = []

            for var_idx in range(num_vars):
                pit_param = pit_values_per_param[var_idx]
                empirical_coverage = [( (pit_param >= (1-cl)/2) & (pit_param <= 1-(1-cl)/2) ).mean() for cl in credible_levels]
                calibration_errors.append(np.abs(np.array(empirical_coverage) - credible_levels).mean())
            
            calibration_error = np.mean(calibration_errors)

            e = 1e-8  # small epsilon to avoid division by zero

            # Update accumulators for point-estimate metrics
            mae_sum += np.sum(np.abs(y_true_denorm - posterior_means_denorm), axis=0)
            mse_sum += np.sum((y_true_denorm - posterior_means_denorm)**2, axis=0)
            mare_sum += np.sum(np.abs((y_true_denorm - posterior_means_denorm) / (y_true_denorm + e)), axis=0)
            mre_sum += np.sum((y_true_denorm - posterior_means_denorm) / (y_true_denorm + e), axis=0)
            total_y_true += np.sum(y_true_denorm, axis=0)
            total_y_true_squared += np.sum(y_true_denorm**2, axis=0)
            count += y_true.shape[0]

            for i in range(y_true.shape[0]):
                mass_1 = round(y_true_denorm[i, mass_1_index], 1)
                mass_2 = round(y_true_denorm[i, mass_2_index], 1)
                distance = round(y_true_denorm[i, distance_index], 1)
                inclination = round(y_true_denorm[i, inclination_index], 1)
                spin_1 = round(y_true_denorm[i, spin_1_index], 1)
                spin_2 = round(y_true_denorm[i, spin_2_index], 1)

                error = np.abs(y_true_denorm[i] - posterior_means_denorm[i])
                sq_error = error ** 2
                per_mass_1_mae[mass_1] += error
                per_mass_1_sq[mass_1] += sq_error 
                per_mass_1_count[mass_1] += 1
                
                per_mass_2_mae[mass_2] += error
                per_mass_2_sq[mass_2] += sq_error
                per_mass_2_count[mass_2] += 1
                
                per_distance_mae[distance] += error
                per_distance_sq[distance] += sq_error
                per_distance_count[distance] += 1

                per_inclination_mae[inclination] += error
                per_inclination_sq[inclination] += sq_error
                per_inclination_count[inclination] += 1

                per_spin_1_mae[spin_1] += error
                per_spin_1_sq[spin_1] += sq_error
                per_spin_1_count[spin_1] += 1

                per_spin_2_mae[spin_2] += error
                per_spin_2_sq[spin_2] += sq_error
                per_spin_2_count[spin_2] += 1

    # Calculate final point-estimate metrics
    mae = mae_sum / count
    mse = mse_sum / count
    rmse = np.sqrt(mse)
    mare = mare_sum / count
    mape = mare*100
    mre = mre_sum / count
    mpe = mre*100
    
    r2 = 1 - (mse_sum / (total_y_true_squared - (total_y_true ** 2 / count)))

    total_mae_loss = np.sum(mae)/num_vars
    
    avg_nll = total_nll / count

    # Per-Mass_1 average MAE
    mass_1_values = sorted(per_mass_1_mae.keys())
    mass_1_mae_values = []
    mass_1_std_values = []
    
    # Per-Mass_2 average MAE
    mass_2_values = sorted(per_mass_2_mae.keys())
    mass_2_mae_values = []
    mass_2_std_values = []
    
    # Per-distance average MAE
    distance_values = sorted(per_distance_mae.keys())
    distance_mae_values = []
    distance_std_values = []

    # Per-inclination average MAE
    inclination_values = sorted(per_inclination_mae.keys())
    inclination_mae_values = []
    inclination_std_values = []

    # Per-spin_1 average MAE
    spin_1_values = sorted(per_spin_1_mae.keys())
    spin_1_mae_values = []
    spin_1_std_values = []

    # Per-spin_2 average MAE
    spin_2_values = sorted(per_spin_2_mae.keys())
    spin_2_mae_values = []
    spin_2_std_values = []

    for m1 in mass_1_values:
        mae_m1 = per_mass_1_mae[m1] / per_mass_1_count[m1]
        mass_1_mae_values.append(mae_m1.tolist())
        std_m1 = np.sqrt((per_mass_1_sq[m1] / per_mass_1_count[m1]) - (mae_m1**2))
        mass_1_std_values.append(std_m1.tolist())
        
    for m2 in mass_2_values:
        mae_m2 = per_mass_2_mae[m2] / per_mass_2_count[m2]
        mass_2_mae_values.append(mae_m2.tolist())
        std_m2 = np.sqrt((per_mass_2_sq[m2] / per_mass_2_count[m2]) - (mae_m2**2))
        mass_2_std_values.append(std_m2.tolist())
        
    for d in distance_values:
        mae_d = per_distance_mae[d] / per_distance_count[d]
        distance_mae_values.append(mae_d.tolist()) # total MAE over all parameters
        std_d = np.sqrt((per_distance_sq[d] / per_distance_count[d]) - (mae_d**2))
        distance_std_values.append(std_d.tolist())

    for i in inclination_values:
        mae_i = per_inclination_mae[i] / per_inclination_count[i]
        inclination_mae_values.append(mae_i.tolist())
        std_i = np.sqrt((per_inclination_sq[i] / per_inclination_count[i]) - (mae_i**2))
        inclination_std_values.append(std_i.tolist())

    for s1 in spin_1_values:
        mae_s1 = per_spin_1_mae[s1] / per_spin_1_count[s1]
        spin_1_mae_values.append(mae_s1.tolist())
        std_s1 = np.sqrt((per_spin_1_sq[s1] / per_spin_1_count[s1]) - (mae_s1**2))
        spin_1_std_values.append(std_s1.tolist())

    for s2 in spin_2_values:
        mae_s2 = per_spin_2_mae[s2] / per_spin_2_count[s2]
        spin_2_mae_values.append(mae_s2.tolist())
        std_s2 = np.sqrt((per_spin_2_sq[s2] / per_spin_2_count[s2]) - (mae_s2**2))
        spin_2_std_values.append(std_s2.tolist())
    
    pit_values_per_param_ = np.array([pv.tolist() for pv in pit_values_per_param])

    num_params = len(pit_values_per_param_)
    p_values = []

    for i, pit_vals in enumerate(pit_values_per_param_):
        sorted_pit = np.sort(pit_vals)
        empirical_cdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)
    
        # KS test against uniform
        stat, p = kstest(pit_vals, 'uniform')
        p_values.append(p)
    
    # Combined p-value using Fisher's method
    combined_stat, combined_p = combine_pvalues(p_values, method='fisher')
    print(f'\nIndividual p-values: {p_values}')
    print(f'Combined p-value (Fisher): {combined_p}')

    metrics = {
        'model_name': model.__class__.__name__,
        'total_mae_loss': total_mae_loss.tolist(),
        'mae': mae.tolist(),
        'mape': mape.tolist(),
        'mare': mare.tolist(),
        'mre': mre.tolist(),
        'mpe': mpe.tolist(),
        'rmse': rmse.tolist(),
        'mse': mse.tolist(),
        'r2': r2.tolist(),
        'average_nll': avg_nll,
        'crps': crps_scores,
        'mean_crps': float(np.mean(crps_scores)),
        'sharpness': sharpness.tolist(),
        'mean_sharpness': float(np.mean(sharpness)),
        'pit_values': pit_values.tolist(),
        'pit_values_per_param': [pv.tolist() for pv in pit_values_per_param],
        'individual_p_values': p_values,
        'combined_p_value': float(combined_p),
        'global_PIT_bias': float(global_PIT_bias),
        'calibration_error': float(calibration_error),
        'mass_1_vs_mae': {
            'mass_1_values': mass_1_values,
            'mae_values': mass_1_mae_values,
            'std_values': mass_1_std_values 
        },
        'mass_2_vs_mae': {
            'mass_2_values': mass_2_values,
            'mae_values': mass_2_mae_values,
            'std_values': mass_2_std_values
        },
        'mae_vs_distance': {
            'distance_values': distance_values,
            'mae_values': distance_mae_values,
            'std_values': distance_std_values
        },
        'mae_vs_inclination': {
            'inclination_values': inclination_values,
            'mae_values': inclination_mae_values,
            'std_values': inclination_std_values
        },
        'mae_vs_spin_1': {
            'spin_1_values': spin_1_values,
            'mae_values': spin_1_mae_values,
            'std_values': spin_1_std_values
        },
        'mae_vs_spin_2': {
            'spin_2_values': spin_2_values,
            'mae_values': spin_2_mae_values,
            'std_values': spin_2_std_values
        }
    }
    
    
    # Save metrics as JSON
    with open(f'{destination_folder}/{model.__class__.__name__}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f'Total mae loss: {total_mae_loss}')
    print(f'Average Negative Log Likelihood: {avg_nll}')

    return metrics

metrics = evaluate_model(model, test_loader, num_vars)


# with open(f'{destination_folder}/{model.__class__.__name__}_metrics.json', 'r') as f:
#     metrics = json.load(f)
    
mass_1_values = metrics['mass_1_vs_mae']['mass_1_values']
mass_1_mae_values = metrics['mass_1_vs_mae']['mae_values']
mass_1_std_values = metrics['mass_1_vs_mae']['std_values']
mass_2_values = metrics['mass_2_vs_mae']['mass_2_values']
mass_2_mae_values = metrics['mass_2_vs_mae']['mae_values']
mass_2_std_values = metrics['mass_2_vs_mae']['std_values']
distance_values = metrics['mae_vs_distance']['distance_values']
distance_mae_values = metrics['mae_vs_distance']['mae_values']
distance_std_values = metrics['mae_vs_distance']['std_values']
inclination_values = metrics['mae_vs_inclination']['inclination_values']
inclination_mae_values = metrics['mae_vs_inclination']['mae_values']
inclination_std_values = metrics['mae_vs_inclination']['std_values']
spin_1_values = metrics['mae_vs_spin_1']['spin_1_values']
spin_1_mae_values = metrics['mae_vs_spin_1']['mae_values']
spin_1_std_values = metrics['mae_vs_spin_1']['std_values']
spin_2_values = metrics['mae_vs_spin_2']['spin_2_values']
spin_2_mae_values = metrics['mae_vs_spin_2']['mae_values']
spin_2_std_values = metrics['mae_vs_spin_2']['std_values']


mean_mass_1_mae_values = [np.mean(mae) for mae in mass_1_mae_values]
mean_mass_1_std_values = [np.mean(std) for std in mass_1_std_values]
mean_mass_2_mae_values = [np.mean(mae) for mae in mass_2_mae_values]
mean_mass_2_std_values = [np.mean(std) for std in mass_2_std_values]
mean_distance_mae_values = [np.mean(mae) for mae in distance_mae_values]
mean_distance_std_values = [np.mean(std) for std in distance_std_values]
mean_inclination_mae_values = [np.mean(mae) for mae in inclination_mae_values]
mean_inclination_std_values = [np.mean(std) for std in inclination_std_values]
mean_spin_1_mae_values = [np.mean(mae) for mae in spin_1_mae_values]
mean_spin_1_std_values = [np.mean(std) for std in spin_1_std_values]
mean_spin_2_mae_values = [np.mean(mae) for mae in spin_2_mae_values]
mean_spin_2_std_values = [np.mean(std) for std in spin_2_std_values]

pit_values_per_param = np.array(metrics['pit_values_per_param'])

num_params = len(pit_values_per_param)


# Plot P-P curve
plt.figure()
p_values = []

param_names = ['m1', 'm2', 'distance', 'inclination', 'spin1', 'spin2']

for i, pit_vals in enumerate(pit_values_per_param):
    sorted_pit = np.sort(pit_vals)
    empirical_cdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)
    
    # KS test against uniform
    stat, p = kstest(pit_vals, 'uniform')
    p_values.append(p)
    
    plt.plot(sorted_pit, empirical_cdf, label=f'{param_names[i]} (p={p:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.xlabel('p')
plt.ylabel('CDF(p)')
plt.title('P-P Curve')
plt.legend()
plt.grid()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_pp_curve.png', bbox_inches='tight', dpi=300)

# Combined p-value using Fisher's method
combined_stat, combined_p = combine_pvalues(p_values, method='fisher')
print(f'\nIndividual p-values: {p_values}')
print(f'Combined p-value (Fisher): {combined_p}')

# Plot MAE vs Mass 1
plt.figure()
plt.plot(mass_1_values, mean_mass_1_mae_values, 'o', markersize=3, linestyle='-')
# plt.errorbar(
#     mass_1_values,               # x
#     mean_mass_1_mae_values,           # y
#     yerr=mean_mass_1_std_values,      # std dev as error bars
#     fmt='o-',                      # 'o' markers and '-' lines
#     markersize=3,
#     capsize=3                      # cap size for the error bars
# )
plt.xlabel('Mass 1 (M_sun)')
plt.ylabel('MAE (average over parameters)')
plt.title('Model Loss vs Mass 1')
plt.grid()
# plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_mae_vs_mass_1.png', bbox_inches='tight', dpi=300)

# Plot MAE vs Mass 2
plt.figure()
plt.plot(mass_2_values, mean_mass_2_mae_values, 'o', markersize=3, linestyle='-')
# plt.errorbar(
#     mass_2_values,               # x
#     mean_mass_2_mae_values,           # y
#     yerr=mean_mass_2_std_values,      # std dev as error bars
#     fmt='o-',                      # 'o' markers and '-' lines
#     markersize=3,
#     capsize=3                      # cap size for the error bars
# )
plt.xlabel('Mass 2 (M_sun)')
plt.ylabel('MAE (average over parameters)')
plt.title('Model Loss vs Mass 2')
plt.grid()
# plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_mae_vs_mass_2.png', bbox_inches='tight', dpi=300) 

# Plot MAE vs Distance
plt.figure()
plt.plot(distance_values, mean_distance_mae_values, 'o', markersize=3, linestyle='-')
# plt.errorbar(
#     distance_values,               # x
#     mean_distance_mae_values,           # y
#     yerr=mean_distance_std_values,      # std dev as error bars
#     fmt='o-',                      # 'o' markers and '-' lines
#     markersize=3,
#     capsize=3                      # cap size for the error bars
# )
plt.xlabel('Distance (Mpc)')
plt.ylabel('MAE (average over parameters)')
plt.title('Model Loss vs Distance')
plt.grid()
# plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_mae_vs_distance.png', bbox_inches='tight', dpi=300)

# Plot MAE vs Inclination
plt.figure()
plt.plot(inclination_values, mean_inclination_mae_values, 'o', markersize=3, linestyle='-')
# plt.errorbar(
#     inclination_values,               # x
#     mean_inclination_mae_values,           # y
#     yerr=mean_inclination_std_values,      # std dev as error bars
#     fmt='o-',                      # 'o' markers and '-' lines
#     markersize=3,
#     capsize=3                      # cap size for the error bars
# )
plt.xlabel('Inclination (degrees)')
plt.ylabel('MAE (average over parameters)')
plt.title('Model Loss vs Inclination')
plt.grid()
# plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_mae_vs_inclination.png', bbox_inches='tight', dpi=300)

# Plot MAE vs Spin 1
plt.figure()
plt.plot(spin_1_values, mean_spin_1_mae_values, 'o', markersize=3, linestyle='-')
# plt.errorbar(
#     spin_1_values,               # x
#     mean_spin_1_mae_values,           # y
#     yerr=mean_spin_1_std_values,      # std dev as error bars
#     fmt='o-',                      # 'o' markers and '-' lines
#     markersize=3,
#     capsize=3                      # cap size for the error bars
# )
plt.xlabel('Spin 1 (degrees)')
plt.ylabel('MAE (average over parameters)')
plt.title('Model Loss vs Spin 1')
plt.grid()
# plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_mae_vs_spin_1.png', bbox_inches='tight', dpi=300)

# Plot MAE vs Spin 2
plt.figure()
plt.plot(spin_2_values, mean_spin_2_mae_values, 'o', markersize=3, linestyle='-')
# plt.errorbar(
#     spin_2_values,               # x
#     mean_spin_2_mae_values,           # y
#     yerr=mean_spin_2_std_values,      # std dev as error bars
#     fmt='o-',                      # 'o' markers and '-' lines
#     markersize=3,
#     capsize=3                      # cap size for the error bars
# )
plt.xlabel('Spin 2 (degrees)')
plt.ylabel('MAE (average over parameters)')
plt.title('Model Loss vs Spin 2')
plt.grid()
# plt.legend()
plt.savefig(f'{destination_folder}/{model.__class__.__name__}_mae_vs_spin_2.png', bbox_inches='tight', dpi=300)

# Print the evaluation metrics
print()
print('-' * 50)
print("Mean Absolute Error (MAE):")
print(f"\tMass 1: {metrics['mae'][0]}")
print(f"\tMass 2: {metrics['mae'][1]}")
print(f"\tDistance: {metrics['mae'][2]}")
print(f"\tInclination: {metrics['mae'][3]}")
print(f"\tSpin 1: {metrics['mae'][4]}")
print(f"\tSpin 2: {metrics['mae'][5]}")

print()
print('-' * 50)
print("Mean Absolute Percentage Error (MAPE):")
print(f"\tMass 1: {metrics['mape'][0]}")
print(f"\tMass 2: {metrics['mape'][1]}")
print(f"\tDistance: {metrics['mape'][2]}")
print(f"\tInclination: {metrics['mape'][3]}")
print(f"\tSpin 1: {metrics['mape'][4]}")
print(f"\tSpin 2: {metrics['mape'][5]}")

print()
print('-' * 50)
print("Mean Relative Error (MRE):")
print(f"\tMass 1: {metrics['mre'][0]}")
print(f"\tMass 2: {metrics['mre'][1]}")
print(f"\tDistance: {metrics['mre'][2]}")
print(f"\tInclination: {metrics['mre'][3]}")
print(f"\tSpin 1: {metrics['mre'][4]}")
print(f"\tSpin 2: {metrics['mre'][5]}")

print()
print('-' * 50)
print("Mean Percentage Error (MPE):")
print(f"\tMass 1: {metrics['mpe'][0]}")
print(f"\tMass 2: {metrics['mpe'][1]}")
print(f"\tDistance: {metrics['mpe'][2]}")
print(f"\tInclination: {metrics['mpe'][3]}")
print(f"\tSpin 1: {metrics['mpe'][4]}")
print(f"\tSpin 2: {metrics['mpe'][5]}")

print()
print('-' * 50)
print("Root Mean Squared Error (RMSE):")
print(f"\tMass 1: {metrics['rmse'][0]}")
print(f"\tMass 2: {metrics['rmse'][1]}")
print(f"\tDistance: {metrics['rmse'][2]}")
print(f"\tInclination: {metrics['rmse'][3]}")
print(f"\tSpin 1: {metrics['rmse'][4]}")
print(f"\tSpin 2: {metrics['rmse'][5]}")

print()
print('-' * 50)
print("Mean Squared Error (MSE):")
print(f"\tMass 1: {metrics['mse'][0]}")
print(f"\tMass 2: {metrics['mse'][1]}")
print(f"\tDistance: {metrics['mse'][2]}")
print(f"\tInclination: {metrics['mse'][3]}")
print(f"\tSpin 1: {metrics['mse'][4]}")
print(f"\tSpin 2: {metrics['mse'][5]}")

print()
print('-' * 50)
print("R^2 Score:")
print(f"\tMass 1: {metrics['r2'][0]}")
print(f"\tMass 2: {metrics['r2'][1]}")
print(f"\tDistance: {metrics['r2'][2]}")
print(f"\tInclination: {metrics['r2'][3]}")
print(f"\tSpin 1: {metrics['r2'][4]}")
print(f"\tSpin 2: {metrics['r2'][5]}")

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
    
    mean = np.load(path + 'Data/lmdb_parameters/mean.npy')
    std = np.load(path + 'Data/lmdb_parameters/std.npy')
    
    # Ensure lines_to_print is a set for efficient membership testing
    lines_to_print = set(range(start_line, end_line + 1))
    
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for inference
        for x_batch, y_batch in dataloader:
            # Move to appropriate device if necessary
            x_batch = x_batch.to(next(model.parameters()).device)
            y_batch = y_batch.to(next(model.parameters()).device)
            
            # Sample from flow posterior to get point estimates (mean)
            samples = []
            samples_per_input = 100   # 1000 is better for P-P plots but slow
            samples_per_batch = 10
            num_batches = samples_per_input // samples_per_batch

            for _ in range(num_batches):  # batch sampling for speed
                s = model(x_batch, parameters=None, num_samples=samples_per_batch) # (batch, samples_per_batch, num_vars)
                samples.append(s)
                
            samples = torch.cat(samples, dim=1) # (batch, num_batches*samples_per_batch, num_vars)
            
            samples = samples.cpu().numpy()  # Convert to numpy for further processing
            posterior_means = samples.mean(axis=1)
            y_true = y_batch.cpu().numpy()

            # Denormalize
            y_true_denorm = y_true*std + mean
            posterior_means_denorm = posterior_means*std + mean
            
            # Print the predictions for the specified lines
            print('\nPrinting predictions...')
            
            for i in range(y_true.shape[0]):
                if global_index in lines_to_print:
                    print(f"Line {global_index}: True values: {y_true_denorm[i]} || Predicted values: {posterior_means_denorm[i]}")
                    printed_count += 1

                    # Stop if all specified lines have been printed
                    if printed_count == len(lines_to_print):
                        return

                global_index += 1
                
# Specify the lines you want to print
start_line = 0
end_line = 20

# Call the prediction_viewer function
prediction_viewer(model, test_loader, start_line, end_line)
