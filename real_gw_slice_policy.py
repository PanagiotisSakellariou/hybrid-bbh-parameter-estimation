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
from collections import defaultdict
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
seed = 99
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

# Deep model
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

# Hybrid Deep Model with CNN and Transformer Encoder
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
        
                
# Training function
def train_model(model, train_loader, val_loader, epochs, resume_checkpoint=None):
    
    if fine_tune:
        print("\nFine-tuning mode enabled. Loading pretrained weights...")
        # Optionally load best pretrained weights (excluding last layer)
        if os.path.exists(weights_path):
            
            print(f'\nLoading best pretrained weights from {weights_path}...')
            pretrained_dict = torch.load(weights_path, map_location=device)
            model_dict = model.state_dict()
            
            # Load everything except the final output layer
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc3" not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("Best pretrained weights loaded successfully (excluding final layer).")
            
            if fine_tune_all:
                print("\nFine-tuning all layers...")
                for param in model.parameters():
                    param.requires_grad = True
            else:
                print("\nFine-tuning only the last layer...")
                for name, param in model.named_parameters():
                    if "fc3" not in name:
                        param.requires_grad = False
        else:
            print(f'\nNo pretrained weights found at {weights_path}. Training from scratch.') 
                      
        model.to(device)
        
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
                start_epoch = checkpoint['epoch']+1
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
            
            running_loss += loss.item()* inputs.size(0)  # Accumulate loss
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


def evaluate_model(model, test_loader, num_vars):
    # Set model to evaluation mode
    model.eval()
    
    # Global metric accumulators
    mae_sum = np.zeros(num_vars)
    mse_sum = np.zeros(num_vars)
    mare_sum = np.zeros(num_vars)
    mre_sum = np.zeros(num_vars)
    total_y_true = np.zeros(num_vars)
    total_y_true_squared = np.zeros(num_vars)
    count = 0
    all_predictions = []

    
    # import parameters from csv
    df = pd.read_csv(csv_path)
    df['keys'] = df['row_key'].astype(str).apply(lambda x: x.split('_', 1)[1])
    df.set_index('keys', inplace=True)
    
    # Per-value accumulators
    mass_1_index = 0  # Mass 1 is the first variable (index 0)
    mass_2_index = 1  # Mass 2 is the second variable (index 1)
    distance_index = 2  # Distance is the third variable (index 2)
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            test_predictions = model(inputs)

            y_true = targets.cpu().numpy().astype(np.float64)
            y_pred = test_predictions.cpu().numpy().astype(np.float64)

            # Adjust predictions based on the parameters from the CSV file
            keys = [f'{m1:.2f}_{m2:.2f}_{d:.1f}' for m1, m2, d in zip(y_true[:, 0].round(2), y_true[:, 1].round(2), y_true[:, 2].round(1))]

            df_batch = df[df.index.isin(keys)].reindex(keys)
            # print(df_batch)  # Debugging line to check the batch DataFrame
            m1_low = df_batch['mass_1'].values + df_batch['mass_1_low'].values
            m1_high = df_batch['mass_1'].values + df_batch['mass_1_high'].values
            m2_low = df_batch['mass_2'].values + df_batch['mass_2_low'].values
            m2_high = df_batch['mass_2'].values + df_batch['mass_2_high'].values
            distance_low = df_batch['distance'].values + df_batch['distance_low'].values
            distance_high = df_batch['distance'].values + df_batch['distance_high'].values

            pr_mass1 = y_pred[:, mass_1_index]
            pr_mass2 = y_pred[:, mass_2_index]
            pr_distance = y_pred[:, distance_index]

            in_m1 = (pr_mass1 >= m1_low) & (pr_mass1 <= m1_high)
            in_m2 = (pr_mass2 >= m2_low) & (pr_mass2 <= m2_high)
            in_distance = (pr_distance >= distance_low) & (pr_distance <= distance_high)
            # in_all = in_m1 & in_m2 & in_distance
            
            adjusted_pred = y_pred.copy()
            # adjusted_pred[in_all] = y_true[in_all]  # Adjust predictions to match true values where conditions are met
            adjusted_pred[in_m1, mass_1_index] = y_true[in_m1, mass_1_index]
            adjusted_pred[in_m2, mass_2_index] = y_true[in_m2, mass_2_index]
            adjusted_pred[in_distance, distance_index] = y_true[in_distance, distance_index]
            
            # Debugging line to check adjusted predictions
            print(m1_low[:10], m1_high[:10], m2_low[:10], m2_high[:10], distance_low[:10], distance_high[:10])  # Debugging line to check parameter ranges
            print(f'True values: {y_true[:10]}')  # Debugging line to check true values
            print(f'Predictions: {y_pred[:10]}')  # Debugging line to check initial predictions
            print(f'Adjusted predictions: {adjusted_pred[:10]}')  # Debugging line to check adjusted predictions

            
            df_new = pd.DataFrame({
                'row_key': df_batch['row_key'].values,
                'event_name': df_batch['event_name'].values,
                'mass_1': y_true[:, mass_1_index],
                'mass_1_lower': df_batch['mass_1_low'].values,
                'mass_1_upper': df_batch['mass_1_high'].values,
                'mass_2': y_true[:, mass_2_index],
                'mass_2_lower': df_batch['mass_2_low'].values,
                'mass_2_upper': df_batch['mass_2_high'].values,
                'distance': y_true[:, distance_index],
                'distance_lower': df_batch['distance_low'].values,
                'distance_upper': df_batch['distance_high'].values,
                'real_pred_mass_1': y_pred[:, mass_1_index],
                'lower_limit_mass_1': m1_low,
                'upper_limit_mass_1': m1_high,
                'adjusted_pred_mass_1': adjusted_pred[:, mass_1_index],
                'real_pred_mass_2': y_pred[:, mass_2_index],
                'lower_limit_mass_2': m2_low,
                'upper_limit_mass_2': m2_high,
                'adjusted_pred_mass_2': adjusted_pred[:, mass_2_index],
                'real_pred_distance': y_pred[:, distance_index],
                'lower_limit_distance': distance_low,
                'upper_limit_distance': distance_high,
                'adjusted_pred_distance': adjusted_pred[:, distance_index]
            })

            all_predictions.append(df_new)
            
            e = 1e-8  # small epsilon to avoid division by zero

            # Global accumulators
            mae_sum += np.sum(np.abs(y_true - adjusted_pred), axis=0)
            mse_sum += np.sum((y_true - adjusted_pred) ** 2, axis=0)
            mare_sum += np.sum(np.abs((y_true - adjusted_pred) / (y_true + e)), axis=0)
            mre_sum += np.sum((y_true - adjusted_pred) / (y_true + e), axis=0)
            total_y_true += np.sum(y_true, axis=0)
            total_y_true_squared += np.sum(y_true**2, axis=0)
            count += y_true.shape[0]


    # Final metrics
    mae = mae_sum / count
    mse = mse_sum / count
    rmse = np.sqrt(mse)
    mare = mare_sum / count
    mape = mare * 100
    mre = mre_sum / count
    mpe = mre * 100

    r2 = 1 - (mse_sum / (total_y_true_squared - (total_y_true ** 2 / count)))
    total_mae_loss = np.sum(mae) / num_vars

        
    final_df = pd.concat(all_predictions, ignore_index=True)
    final_df.to_csv(f'{destination_folder}/{model.__class__.__name__}_real_predictions.csv', index=False)

    # Output dictionary
    metrics = {
        'model_name': model.__class__.__name__,
        'loss': total_mae_loss.tolist(),
        'mae': mae.tolist(),
        'mape': mape.tolist(),
        'mare': mare.tolist(),
        'mre': mre.tolist(),
        'mpe': mpe.tolist(),
        'rmse': rmse.tolist(),
        'mse': mse.tolist(),
        'r2': r2.tolist(),      
    }
    

    # # Save metrics as JSON
    with open(f'{destination_folder}/{model.__class__.__name__}_real_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics

slice_names = ['slice1', 'slice2', 'slice3', 'slice4', 'slice5',
               'slice1_reverse', 'slice2_reverse', 'slice3_reverse', 'slice4_reverse', 'slice5_reverse']

# Initialize an empty DataFrame to store summary results
results_df = pd.DataFrame(columns=[
    'model_name', 'slice_name', 'best_training_loss', 'best_train_epoch',
    'best_validation_loss', 'best_val_epoch', 'boundary_test_loss'
])

for model_name in ['DeepModel', 'HybridDeepModel']:
    for slice_name in slice_names:
        path = '/home/sakellariou/hero_disk/test/'  # <--- Change this to the path of the dataset
        csv_path = path + 'real_gw.csv'

        real_train_dataset = IterTensorFlowToPyTorchDataset(path + f'{slice_name}_train_dataset')
        print('\nNumber of training samples in real dataset:', len(real_train_dataset))
        print(f'\nSize of real training dataset: {real_train_dataset.size(in_gb=False):.3f} MB')

        real_test_dataset = IterTensorFlowToPyTorchDataset(path + f'{slice_name}_test_dataset')
        print('\nNumber of test samples in real dataset:', len(real_test_dataset))
        print(f'\nSize of real test dataset: {real_test_dataset.size(in_gb=False):.3f} MB')

        # Create DataLoader for batching
        print('\nCreating DataLoaders...')
        batch = 1024
        real_train_loader = DataLoader(real_train_dataset, batch_size=batch, shuffle=False)
        real_test_loader = DataLoader(real_test_dataset, batch_size=batch, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        num_vars = 3
        models = {
            'DeepModel': DeepModel,
            'HybridDeepModel': HybridDeepModel
        }

        model = models[model_name](num_vars).to(device)
        
        criterion = nn.L1Loss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # fine tuning all layers shows the adaptation capability of the model on new data but not purely generalization.
        # fine tuning only the last layer shows how well the features learned generalize to new data.
        fine_tune= True # Set to True to enable fine-tuning or False to disable
        fine_tune_all = True  # Set to True to fine-tune all layers or False to fine-tune only the last layer, if fine_tune is enabled


        weights_path = f'./{model.__class__.__name__}_results/{model.__class__.__name__}_best_model.pth' # <--- Change the path to the pretrained weights if needed

        ## Create the scheduler, monitoring validation loss
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-5)

        early_stop = False         # Set to True to enable early stopping or False to disable
        early_stop_patience = 50

        # Print model's name
        print(f'\nModel : {model.__class__.__name__}')
        print(f'\nOptimizer: {optimizer.__class__.__name__} || Initial learning rate: {optimizer.param_groups[0]["lr"]}')

        # Create the destination folder if it doesn't exist
        destination_folder = f'./real_gw_slice_analysis/{slice_name}_{model.__class__.__name__}_results' # <--- Change this to your desired destination folder
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        print(f'\nResults will be saved in: {destination_folder}\n')
        
        # Train the model
        epochs = 1
        checkpoint = None #f'{destination_folder}/{model.__class__.__name__}_checkpoint.pth'
        train_losses, val_losses = train_model(model, real_train_loader, real_test_loader, epochs, resume_checkpoint=checkpoint)
        
        # Best training and validation losses and their epochs
        best_train_loss = min(train_losses)
        best_train_epoch = train_losses.index(best_train_loss) + 1
        
        best_val_loss = min(val_losses)
        best_val_epoch = val_losses.index(best_val_loss) + 1
        
        
        
        # Plot the training and validation losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model.__class__.__name__} Loss')
        plt.grid()
        plt.legend()
        plt.savefig(f'{destination_folder}/{model.__class__.__name__}_loss.png', dpi=300, bbox_inches='tight')
        
        # Load the best model fine_tuned weights
        print('\nLoading best model weights...')
        try:
            model.load_state_dict(torch.load(f'{destination_folder}/{model.__class__.__name__}_best_model.pth'))
            print('\nModel weights loaded successfully.')
        except:
            print('\nError loading model weights. Initializing model...')
            with torch.no_grad():
                inputs, targets = next(iter(real_train_loader))
                inputs = inputs.to(device)
                model.initialize(inputs)
            model.load_state_dict(torch.load(f'{destination_folder}/{model.__class__.__name__}_best_model.pth'))
            print('\nModel weights loaded successfully.')
            
        # Make more evaluation metrics
        print()
        print('Making evaluation metrics...')
        metrics = evaluate_model(model, real_test_loader, 3)
        print(f'\nTest Loss: {metrics["loss"]}')
        
        boundary_test_loss = metrics["loss"]
        
        results_df = pd.concat([
            results_df,
            pd.DataFrame([{
                'model_name': model.__class__.__name__,
                'slice_name': slice_name,
                'best_training_loss': best_train_loss,
                'best_train_epoch': best_train_epoch,
                'best_validation_loss': best_val_loss,
                'best_val_epoch': best_val_epoch,
                'boundary_test_loss': boundary_test_loss
            }])
        ], ignore_index=True)

results_df.to_csv('./real_gw_slice_analysis/real_gw_slice_summary.csv', index=False)
print('\nSummary results saved to real_gw_slice_summary.csv')