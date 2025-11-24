# make tensorflow datasets from lmdb with real gw
from lmdb_dataset_handler import LMDBDataset
import tensorflow as tf


path ='/home/sakellariou/hero_disk/test/'  # <---- Change path here

data_filename = path + 'real_gw.lmdb'      # <---- Change the name of the lmdb file here
csv_filename = path + 'real_gw.csv'        # <---- Change the name of the csv file here

database = LMDBDataset(data_filename)
print(f'\nLMDB size: {database.size(in_gb=True)} GB')

keys = database.get_keys()
print(f'Number of keys in the database: {len(keys)}')

# train_keys, val_keys, test_keys = database.split_keys()

# choose keys based on the parameters our neural network is trained on 
filtered_keys = []
for key in keys:
    _, m1, m2, d = key.split('_')
    if float(m1) >= 5 and float(m2) >= 5 and float(d) <= 500 and float(d) >= 50 and float(m1) <= 100 and float(m2) <= 100:
        filtered_keys.append(key)
print(f'Number of keys after filtering: {len(filtered_keys)}')
print(filtered_keys)


if len(filtered_keys) < 2:
    raise ValueError("Not enough keys to create train/test splits.")


# Generate cases dynamically
cases = []
slices = []

# Normal slices: train = [i], test = all except i
for i in range(len(filtered_keys)):
    cases.append(f"slice{i+1}")
    slices.append(([filtered_keys[i]], filtered_keys[:i] + filtered_keys[i+1:]))

# Reverse slices: train = all except i, test = [i]
for i in range(len(filtered_keys)):
    cases.append(f"slice{i+1}_reverse")
    slices.append((filtered_keys[:i] + filtered_keys[i+1:], [filtered_keys[i]]))

print(f"Total cases: {len(cases)}")


for case, (train_keys, test_keys) in zip(cases, slices):
    
    print(f'\nProcessing case: {case}')
    print(f'Train keys: {train_keys}')
    print(f'Test keys: {test_keys}')

    print(f'Number of training keys: {len(train_keys)}')
    print(f'Number of test keys: {len(test_keys)}')

    # Extract minimum and maximum values from the training set for min-max normalization
    print(f'\nExtracting minimum and maximum values from the training set for normalization...')
    minimum, maximum = database.min_max_extractor(keys=train_keys)
    print('Minimum:', minimum)
    print('Maximum:', maximum)

    # # Create TensorFlow datasets
    print(f'\nCreating TensorFlow datasets...')
    print(f'\nMaking training dataset with {len(train_keys)} elements...')
    train_dataset = database.create_tensorflow_dataset(path, keys=train_keys, minimum=minimum, maximum=maximum)
    tf.data.Dataset.save(train_dataset, path + f'{case}_train_dataset')
    print(f"Train Dataset saved!")

    print(f'\nMaking test dataset with {len(test_keys)} elements...')
    test_dataset = database.create_tensorflow_dataset(path, keys=test_keys, minimum=minimum, maximum=maximum)
    tf.data.Dataset.save(test_dataset, path + f'{case}_test_dataset')
    print(f"Test Dataset saved!")

print(f"\nProcessed {len(cases)} cases with {len(filtered_keys)} filtered keys.")
