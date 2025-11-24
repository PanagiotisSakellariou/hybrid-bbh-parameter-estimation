# this makes the final tensorflow dataset with real noise added to the clean strains from LMDB
# the noisy strains will be min -max normalized based on the min and max values computed from the noisy training strains

import os
import lmdb
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
import shutil
import sys
import pycbc.noise 
import pycbc.psd

# ------------------------------
# Split keys for train/val/test
# ------------------------------
def split_keys(env, test_size=0.3, val_size=0.1, n_keys=None):
    with env.begin() as txn:
        if n_keys is not None:
            print(f"Collecting first {n_keys} keys from LMDB...")
            all_keys = []
            cursor = txn.cursor()
            for i, (key, _) in enumerate(cursor):
                if i >= n_keys:
                    break
                all_keys.append(key.decode())
            print(f"Total keys collected: {len(all_keys)}")
        else:
            print("Collecting all keys from LMDB...")
            all_keys = [key.decode() for key, _ in txn.cursor()]
            print(f"Total keys found: {len(all_keys)}")

        all_keys = np.array(all_keys)
        train_keys, test_keys = train_test_split(all_keys, test_size=test_size, random_state=7)
        train_keys, val_keys = train_test_split(train_keys, test_size=val_size, random_state=1)
        print(f"Train keys: {len(train_keys)}, Validation keys: {len(val_keys)}, Test keys: {len(test_keys)}")

        return train_keys, val_keys, test_keys

# ------------------------------
# Yield strain and parameters in batches
# ------------------------------
def yield_data(strains_env, keys, batch_size=1024):
    with strains_env.begin() as strain_txn:
        batch_strains = []
        batch_parameters = []
        batch_keys = []
        for key in keys:
            strain = np.frombuffer(strain_txn.get(key.encode()), dtype=np.float64)
            parameters = np.array(key.split('_')[1:], dtype=np.float64)
            batch_strains.append(strain)
            batch_parameters.append(parameters)
            batch_keys.append(key)
            
            if len(batch_strains) == batch_size:
                yield np.stack(batch_keys), np.stack(batch_strains), np.stack(batch_parameters)
                batch_strains = []
                batch_parameters = []
                batch_keys = []
        if batch_strains:
            yield batch_keys, np.stack(batch_strains), np.stack(batch_parameters)

# ------------------------------
# Add random noise from LMDB
# ------------------------------
def add_noise(strains_batch):
    noisy_batch = []

    flow = 10.0
    delta_f = 1.0 / 16
    flen = int(8192 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    delta_t = 1.0 / 8192
    tsamples = int(1 / delta_t)
    
    for strain in strains_batch:
        noise = pycbc.noise.noise_from_psd(tsamples, delta_t, psd)
        noise = noise.numpy().astype(np.float64)
        
        if np.isnan(noise).any():
            print("Warning: NaNs detected in noise!")
            
        noisy_strain = np.array(strain + noise, dtype=np.float64)

        noisy_batch.append(noisy_strain)
    
    return np.stack(noisy_batch)

# ------------------------------
# Create temporal Lmdb database with noise added strains
# ------------------------------
def write_log(log_path, message):
    with open(log_path, "a") as logf:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logf.write(f"[{timestamp}] {message}\n")
        
def get_lmdb_size_gb(path):
    if not os.path.exists(path):
        return 0.0
    total_size = 0
    for fname in ["data.mdb", "lock.mdb"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            total_size += os.path.getsize(fpath)
    size_gb = total_size / (1024**3)
    return size_gb
        
def create_temp_lmdb(strains_env, keys, save_path, log_path="noisy_generation.log", batch_size=1024):
    if os.path.exists(save_path):
        print(f'LMDB database already exists at {save_path}.')
        response = input('Do you want to overwrite the existing database? (y/n): ')
        if response.lower() == 'n':
            print('Skipping LMDB database creation.')
            sys.exit()
        else:
            print(f'Deleting existing LMDB database...')
            try:
                shutil.rmtree(save_path)
                if os.path.exist(log_path):
                    os.remove(log_path)
                print(f'Existing LMDB database deleted.')
        
            except OSError as e:
                print("Error deleting existing LMDB database. Skipping creation.")
                print(e)
                sys.exit()
            
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
    noisy_strains_env = lmdb.open(save_path, map_size=int(1e12))
    
    with noisy_strains_env.begin() as txn_out:
        completed_keys = set(key.decode() for key, _ in txn_out.cursor())
        
    remaining_keys = [key for key in keys if key not in completed_keys]
    
    print(f"Creating noisy LMDB database...{len(remaining_keys)} samples to process.")
     
    pbar = tqdm(total=len(remaining_keys), desc="Generating noisy samples", unit="samples", dynamic_ncols=True)
    
    for batch_idx, (keys_batch, strains_batch, _) in enumerate(yield_data(strains_env, remaining_keys, batch_size)):
        
        if np.isnan(strains_batch).any():
                print("Warning: NaNs detected in batch!")
                
        noisy_batch = add_noise(strains_batch)
                
        with noisy_strains_env.begin(write=True) as txn_out:
            for i, noisy_strain in enumerate(noisy_batch):
                key = keys_batch[i].encode()
                txn_out.put(key, noisy_strain.astype(np.float64).tobytes())
        
        write_log(log_path=log_path, message=f"Batch {batch_idx+1}: Processed {len(noisy_batch)} samples: Last key {keys_batch[-1]}")
        pbar.update(len(noisy_batch))
        strain_size = get_lmdb_size_gb(save_path)
        pbar.set_postfix({"Noisy Strain DB": f"{strain_size:.5f} GB"})
        
    pbar.close()
    print("Noisy LMDB database created successfully.")
    
# ------------------------------
# Create TensorFlow dataset
# ------------------------------
def create_noise_tensorflow_dataset(clean_strains_env, noisy_strains_env, keys, save_path, dataset_name, batch_size=1024, compute_min_max=True): 
    os.makedirs(save_path, exist_ok=True)
            
    # Select strain source
    strains_env = noisy_strains_env if compute_min_max else clean_strains_env
    
    # Compute min/max over noisy strains
    if compute_min_max:
        print("Computing min and max for normalization...")

        min_val = np.inf
        max_val = -np.inf
            
        total_samples = len(keys)
        
        pbar = tqdm(total=total_samples, desc="Computing min/max", unit="samples", dynamic_ncols=True)
        for _, strains_batch, _ in yield_data(strains_env, keys, batch_size):
            
            if np.isnan(strains_batch).any():
                print("Warning: NaNs detected in batch!")  
                  
            min_val = min(min_val, np.min(strains_batch))
            max_val = max(max_val, np.max(strains_batch))
            
            pbar.update(len(strains_batch)) # Update progress bar by number of samples in batch
        
        pbar.close()
        print(f"Minimum: {min_val}, Maximum: {max_val}")
        np.save(os.path.join(save_path, 'minimum.npy'), min_val)
        np.save(os.path.join(save_path, 'maximum.npy'), max_val)
    else:
        min_val = np.load(os.path.join(save_path, 'minimum.npy'))
        max_val = np.load(os.path.join(save_path, 'maximum.npy'))
        print(f"Loaded min and max from files: Minimum={min_val}, Maximum={max_val}")

    # TensorFlow generator
    def generator():
        total_samples = len(keys)
        pbar = tqdm(total=total_samples, desc=f"{dataset_name.title()} generation", unit="samples", dynamic_ncols=True)
        for _, strain_batch, parameters_batch in yield_data(strains_env, keys, batch_size):
            # Add noise if using clean strains
            if strains_env == clean_strains_env:
                strain_batch = add_noise(strain_batch)
            # min-max normalization
            norm_batch = (strain_batch - min_val) / (max_val - min_val)
            
            for strain, parameters in zip(norm_batch, parameters_batch):
                yield strain.astype(np.float64), parameters.astype(np.float64)
                pbar.update(1)
        pbar.close()
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(8192,), dtype=tf.float64),
            tf.TensorSpec(shape=(6,), dtype=tf.float64)
        )
    )

    # Save dataset
    tf.data.Dataset.save(dataset, os.path.join(save_path, dataset_name))
    print(f"{dataset_name} saved successfully at {save_path}")

    return dataset

# ------------------------------
# Main function
# ------------------------------
def main():
    base_path = '/home/sakellariou/hero_disk/test/'  #<---- change here to the path where the lmdb with the gravitational wave strains is
    save_path = base_path                            #<---- change here to the path you want to save the final datasets
    grawa_path = os.path.join(base_path, 'grawa.lmdb')
      

    grawa_env = lmdb.open(grawa_path, readonly=True, lock=False)
    

    # Split keys
    n_keys = None  # Set to None to use all keys
    batch = 10000
    train_keys, val_keys, test_keys = split_keys(grawa_env, test_size=0.3, val_size=0.1, n_keys=n_keys)
    
    # Save the keys for later retrieval
    np.save(os.path.join(base_path, 'train_keys.npy'), train_keys)
    np.save(os.path.join(base_path, 'val_keys.npy'), val_keys)
    np.save(os.path.join(base_path, 'test_keys.npy'), test_keys)
    
    
    # train_keys = np.load(os.path.join(base_path, 'train_keys.npy'))
    # val_keys = np.load(os.path.join(base_path, 'val_keys.npy'))
    # test_keys = np.load(os.path.join(base_path, 'test_keys.npy'))

    # Create datasets
    noisy_strains_path = os.path.join(base_path, 'noisy_strains_train.lmdb')       # the path of the temporary lmdb with noisy strains
    noisy_log_path = os.path.join(base_path, 'noisy_strains_train_generation.log')
    create_temp_lmdb(grawa_env, train_keys, noisy_strains_path, log_path=noisy_log_path, batch_size=batch)
    noisy_strains_env = lmdb.open(noisy_strains_path, readonly=True, lock=False)
    train_dataset = create_noise_tensorflow_dataset(grawa_env, noisy_strains_env, train_keys, save_path, 'train_dataset', compute_min_max=True)
    val_dataset = create_noise_tensorflow_dataset(grawa_env, noisy_strains_env, val_keys, save_path, 'val_dataset', compute_min_max=False)
    test_dataset = create_noise_tensorflow_dataset(grawa_env, noisy_strains_env, test_keys, save_path, 'test_dataset', compute_min_max=False)

if __name__ == "__main__":
    main()
