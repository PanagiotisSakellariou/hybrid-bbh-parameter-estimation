# this code helps make things from the LMDB database
# it uses the Dataset class from PyTorch

import numpy as np
import torch
from torch.utils.data import Dataset
import lmdb
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

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
    

torch.set_printoptions(precision=10) # Set the precision for printing tensors



class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, keys=None, minimum=None, maximum=None):
        self.lmdb_path = lmdb_path
        self.keys = keys
        self.minimum = minimum
        self.maximum = maximum
               
        # Open LMDB environment
        self.env = lmdb.open(lmdb_path, readonly=True, create=False, lock=False)

    def __len__(self):
        return self.count_elements(keys=self.keys)
        
    
    def __getitem__(self, index):
        
        if self.keys is None:
            self.keys = self.get_keys()
            
        key = self.keys[index]
        
        if self.minimum is None or self.maximum is None:
            self.minimum, self.maximum = self.min_max_extractor(keys=self.keys)
        
        key, value = next(self.normalizer(self.minimum, self.maximum, keys=[key]))
        
        # Extract features and label
        features = torch.tensor(value, dtype=torch.float32).view(1, -1)  # Reshape to match model input shape
        label = torch.tensor(np.array(key.split('_')[1:], dtype=np.float32), dtype=torch.float32)
        return features, label
        
    
    def yield_data(self, keys=None):
        ''' Reads key-value pairs from an LMDB database and yields them as a generator.
            If keys are None, all key-value pairs are read.
            Args: 
               keys: A list of keys to read from the LMDB database not as byte strings.
            Yields: A tuple of (key, value).
        '''
        with self.env.begin() as txn:
            if keys is None:
                for key, value in txn.cursor():
                    key = key.decode()
                    value = np.frombuffer(value, dtype=np.float64)
                    yield key, value
            else:         
                for key in keys:
                    value = txn.get(key.encode())
                    value = np.frombuffer(value, dtype=np.float64)
                    yield key, value
        
    def count_elements(self, keys=None):
        ''' Count the number of elements in an LMDB database.
            Returns:
                The number of elements in the LMDB database.
        '''
        if keys is None:
            with self.env.begin() as txn:
                count = txn.stat()['entries']
                return count   
        else:
            return len(keys)
        
    def get_keys(self, keys_as_bytes=False):
        ''' Extracts all the keys from an LMDB database.
            Args: 
                keys_as_bytes: If True, the keys are returned as byte strings.
            Returns: A list of keys.
        '''
        with self.env.begin() as txn:
            if keys_as_bytes:
                keys = [key for key, _ in txn.cursor()]
            else:
                keys = [key.decode() for key, _ in txn.cursor()]
            return keys
            
                    
    def split_keys(self, keys=None, test_size=0.3, val_size=0.1, keys_as_bytes=False):
        ''' Extracts keys from an LMDB database and splits 
            them for training, test, and validation.
            Args: 
                keys: A list of keys to read from the LMDB database.
                The keys should not be byte strings. if keys=None, all key-value pairs are read.
                test_size: The proportion of the dataset to include in the test split.
                val_size: The proportion of the dataset to include in the validation split.
                keys_as_bytes: If True, the keys are returned as byte strings.
            Returns: A list of keys for training, validation, and testing.
        '''
        with self.env.begin() as txn:
            if keys is None:
                if keys_as_bytes:
                    all_keys =  [key for key, _ in txn.cursor()]
                else:
                    all_keys =  [key.decode() for key, _ in txn.cursor()]
                    
                all_keys = np.array(all_keys).reshape(-1)
                
                train_keys, test_keys = train_test_split(all_keys, test_size=test_size, random_state=7)
                train_keys, val_keys = train_test_split(train_keys, test_size=val_size, random_state=1)
                return train_keys, val_keys, test_keys
            
            else:
                byte_keys = []
                if keys_as_bytes:
                    for key in keys:
                        byte_keys.append(key.encode())                    
                    train_keys, test_keys = train_test_split(byte_keys, test_size=test_size, random_state=7)
                else:
                    train_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=7)
                
                train_keys, val_keys = train_test_split(train_keys, test_size=val_size, random_state=1)
                return train_keys, val_keys, test_keys
            
           
    def find_value(self, key_, specific_key=False):
        ''' Find the value of a key in the LMDB database.
            Args:
                key: The key to search for in the LMDB database.
                specific_key: If True, the value of this specific key will be returned if exists.
                              If False, a list of all keys that contain the given key name
                              and a list with their values will be returned.                  
            Returns:
                The value of the key.
        '''
        with self.env.begin() as txn:
            if specific_key:
                value = txn.get(key_.encode())
                if value is None:
                    print(f'Key {key_} not found in the LMDB database.')
                    return key_, 'None'
                else:
                    value = np.frombuffer(value, dtype=np.float64)  # Assuming values are stored as float64
                    return key_, value
            else:
                key_ = key_.encode()
                key_list = []
                value_list = []
                for key, value in txn.cursor():
                    if key_ in key:
                        key_list.append(key.decode())
                        value = np.frombuffer(value, dtype=np.float64)
                        value_list.append(value)
                    print(f'Keys found: {len(key_list)}', end='\r', flush=True)   
                if len(key_list) == 0:
                    print(f'Key {key_.decode()} not found in the LMDB database.')
                    return ['None'], ['None']
                else:
                    return key_list, value_list   
                 
    def min_max_extractor(self, keys=None):
        ''' Returns minimum and maximum of the values given a list of keys.
            Args: 
                keys: A list of keys to read from the LMDB database.
                if keys=None, all key-value pairs are read.
        '''
        min_value=np.inf
        max_value=-np.inf
        
        for _, value in self.yield_data(keys=keys):
            
            min_value = np.minimum(min_value, np.min(value))
            max_value = np.maximum(max_value, np.max(value))
        
        return min_value, max_value
    
    def normalizer(self, minimum, maximum, keys=None):
        ''' Min-Max Normalization on the values of the dataset.
            Args:
                minimum: The minimum value of the dataset.
                maximum: The maximum value of the dataset.
                keys: A list of keys to read from the LMDB database.
                if keys=None, all key-value pairs are read.
        '''
        
        for key, value in self.yield_data(keys=keys):
            value = (value - minimum) / (maximum - minimum)
            yield key, value
    
    def mean_std_extractor(self, keys=None):
        """
        Computes mean and standard deviation of values from an LMDB database.

        Args:
            keys: A list of keys to read from the LMDB database.
                If keys=None, all key-value pairs are read.

        Returns:
            mean (np.ndarray), std (np.ndarray)
        """
        if keys is None:
            keys = self.get_keys()

        n = 0
        mean = None
        M2 = None  # Sum of squares of differences from the mean

        for key in keys:
            # Extract the label vector from the key
            labels = np.array(key.split('_')[1:], dtype=np.float64)  # shape: (num_features,)

            batch_n = 1
            batch_mean = labels  # single sample, mean = sample itself
            batch_var = np.zeros_like(labels)  # variance of a single sample = 0

            if mean is None:
                mean = batch_mean
                M2 = batch_var * batch_n  # zeros
                n = batch_n
            else:
                delta = batch_mean - mean
                total_n = n + batch_n

                mean = mean + delta * batch_n / total_n
                M2 = M2 + batch_var * batch_n + (delta ** 2) * n * batch_n / total_n
                n = total_n

        variance = M2 / n
        std = np.sqrt(variance)

        return mean, std

    def zscore_normalizer(self, mean, std, keys=None):
        ''' Z-Score Normalization on the values of the dataset.
            Args:
                mean: Mean value(s) of the dataset (can be scalar or array-like).
                std: Standard deviation value(s) of the dataset (same shape as mean).
                keys: A list of keys to read from the LMDB database.
                    If keys=None, all key-value pairs are read.
        '''
        for key, value in self.yield_data(keys=keys):
            value = (value - mean) / std
            yield key, value
            
    def size(self, in_gb=False):
        ''' Returns the size of the LMDB in MB or GB.
        '''
        data_file = self.lmdb_path + '/data.mdb'
        if os.path.exists(data_file):
            size_in_bytes = os.stat(data_file).st_size
            
            if in_gb == False:
                size_in_mb = size_in_bytes / (1024*1024) # Convert to MB
                return size_in_mb
            else:
                size_in_gb = size_in_bytes / (1024*1024*1024) # Convert to GB
                return size_in_gb
        else:
            raise FileNotFoundError(f"Data file not found at {self.lmdb_path}")
       
    def create_lmdb(self, keys, path):
        ''' Create an LMDB database.
            Args:
                keys: A list of keys to retrieve from the main LMDB database
                and store in a new LMDB database.
        '''
        
        # Check if the environment exists
        if os.path.exists(path):
            print(f'LMDB database already exists at {path}.')
            response = input('Do you want to overwrite the existing database? (y/n): ')
        
            if response.lower() == 'n':
                print('Skipping deleting the LMDB database.')
                return
            else:
                # Delete existing files (with proper error handling)
                print(f'Deleting existing LMDB database...')
                try:
                    # os.remove(path + '/lock.mdb')  # Remove lock file
                    os.remove(path + '/data.mdb')  # Remove data file
                    os.rmdir(path)
                except OSError as e:
                    # Handle potential errors (e.g., file in use)
                    print("Error deleting existing LMDB database. Skipping creation.")
                    print(e)
                
        with self.env.begin() as txn:
            new_env = lmdb.open(path, map_size=int(1e12), lock=False) # 1 TB
            
            chunk_size = 1000
            
            txn_new = new_env.begin(write=True)
                
            for i, key in enumerate(keys):
                value = txn.get(key.encode())
                if value is not None:
                    txn_new.put(key.encode(), value)
                else:
                    print(f'Key {key} not found in the LMDB database.')        

                if (i + 1) % chunk_size == 0: # Commit every chunk_size
                    txn_new.commit()
                    txn_new = new_env.begin(write=True)
                        
            txn_new.commit()
    
    
    def create_tensorflow_dataset(self, path, keys=None, minimum=None, maximum=None):
        ''' Create a TensorFlow dataset from an LMDB database.
            Args:
                keys: A list of keys to read from the LMDB database.
                if keys=None, all key-value pairs are read.
                minimum: The minimum value of the dataset.
                maximum: The maximum value of the dataset.
                                
            Returns:
                A TensorFlow dataset.
        '''
        if keys is None:
            keys = self.get_keys()
            
        total_keys = len(keys)
            
        if minimum is None or maximum is None:
            minimum, maximum = self.min_max_extractor(keys=keys)
        
        def generator():
            
            with tqdm(total=total_keys, desc="Normalizing data", unit="sample") as pbar:

                for key, value in self.normalizer(minimum, maximum, keys=keys):
                    labels = np.array(key.split('_')[1:], dtype=np.float64) 
                    features = value
                    yield features, labels
                    pbar.update(1)
            
        
        
        dataset = tf.data.Dataset.from_generator(generator,
        output_signature=(tf.TensorSpec(shape=(8192,), dtype=tf.float64), tf.TensorSpec(shape=(3,), dtype=tf.float64)))
        
        # tf.data.Dataset.save(dataset, path+'dataset')
        # print(f"Dataset saved!")
        
        return dataset

    
    
                       

class TensorFlowToPyTorchDataset(Dataset):
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
    
    def __getitem__(self, index):
        element = next(iter(self.tf_dataset.skip(index).take(1).as_numpy_iterator()))
        features, labels = element
        features = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32) 
        return features, labels
    
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


                        
def main():    
    
    path = '/home/sakellariou/hero_disk/test/'
    data_filename = path + "strains.lmdb"
        
    data = LMDBDataset(data_filename)

    print(f'LMDB size: {data.size(in_gb=True)} GB')
    
    # Count the number of elements in the LMDB database
    print(f'The LMDB has {data.count_elements()} elements.')
    
    
if __name__ == '__main__':
    main()