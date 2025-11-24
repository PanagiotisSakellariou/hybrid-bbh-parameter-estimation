# make lmdb database with real events and csv for human readability
from pycbc.catalog import Merger, Catalog
from pycbc.types import TimeSeries
import numpy as np
import os
import lmdb
import matplotlib.pyplot as plt
import shutil
import sys
import csv

# %matplotlib inline  
np.set_printoptions(linewidth=np.inf)

# Path to save LMDB
path = '/home/sakellariou/hero_disk/test/real_gw.lmdb'    # <---- Change path here
csv_path = '/home/sakellariou/hero_disk/test/real_gw.csv'


# Ask before overwriting
if os.path.exists(path):
    print(f'LMDB database already exists at {path}.')
    response = input('Do you want to overwrite the existing database? (y/n): ')
    if response.lower() == 'n':
        print('Skipping LMDB database creation.')
        sys.exit()
    else:
        print(f'Deleting existing LMDB database...')
        try:
            shutil.rmtree(path)
            print(f'Existing LMDB database deleted.')
            if os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                    print(f'Existing CSV file deleted.')
                except OSError as e:
                    print(f"Error deleting existing CSV file: {e}")
                    sys.exit()
        except OSError as e:
            print("Error deleting existing LMDB database. Skipping creation.")
            print(e)
            sys.exit()

csv_header = ['row_key', 'event_name', 'mass_1', 'mass_2', 'distance', 'gps_time', 'mass_1_low', 'mass_1_high', 'mass_2_low', 'mass_2_high', 'distance_low', 'distance_high']            
with open(csv_path, 'w', newline='') as csvfile:
    data_writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')  
    data_writer.writerow(csv_header)  # Write the header row
    
         
# Create a new LMDB environment
env = lmdb.open(path, map_size=int(1e12), lock=False)
txn = env.begin(write=True)

chunk_size = 1000  # number of samples before commit
counter = 0  # global event counter

for source in ['gwtc-1', 'gwtc-2', 'gwtc-2.1', 'gwtc-3']:
    catalog = Catalog(source=source)
    print(f"Processing source: {source}")
    event_names = list(catalog.names)
    print(f"Found {len(event_names)} events in {source}")

    for name in event_names:
        event = catalog[name]  # type: Merger

        # Basic info
        t0 = event.time
        m1 = event.mass1
        m2 = event.mass2
        d = event.distance
        m1_low = event._raw_mass_1_source_lower
        m1_high = event._raw_mass_1_source_upper
        m2_low = event._raw_mass_2_source_lower
        m2_high = event._raw_mass_2_source_upper
        d_low = event._raw_luminosity_distance_lower
        d_high = event._raw_luminosity_distance_upper

        print(f"Processing event: {name}, time: {t0}, masses: {m1}, {m2}, distance: {d}")

        try:
            # Try fetching strain from Hanford
            ts = event.strain(ifo='H1', sample_rate=16384, duration=32)
        except Exception as e:
            print(f"Failed to fetch strain data for {name} in H1: {e}")
            continue
        
        ts_numpy = ts.numpy()
        # if NaNs in the strain the event is excluded
        if np.isnan(ts_numpy).any() or np.isinf(ts_numpy).any():
            print(f"Warning: NaNs detected in full strain data for event {name}.")
            continue
            
        fs = ts.sample_rate
        t_start = ts.start_time
        t_end = ts.end_time
        print(f"Sample rate: {fs}, Start time: {t_start}, End time: {t_end}")
        
        # Cut around the merger (±0.7s, +0.3s)
        start_time = t0 - 0.7
        end_time = t0 + 0.3
        try:
            segment = ts.time_slice(start_time, end_time)
            delta_t = 1 / 8192
            segment_resampled = segment.resample(delta_t)
        except Exception as e:
            print(f"Could not slice or resample event {name}: {e}")
            continue

        # Encode key and data
        counter += 1
        row_key = f"{counter:09}_{m1:.2f}_{m2:.2f}_{d:.1f}"
        data_row = segment_resampled.numpy().astype(np.float64)
        row_metadata = [
            row_key,
            name,
            m1, m2, d, t0,
            m1_low, m1_high,
            m2_low, m2_high,
            d_low, d_high
        ]

        # Save to CSV
        with open(csv_path, mode='a', newline='') as csvfile:
            data_writer = csv.writer(csvfile, delimiter=",", lineterminator="\n")
            data_writer.writerow(row_metadata)
        # Put into LMDB
        txn.put(row_key.encode(), data_row.tobytes())

        # Commit every chunk_size
        if counter % chunk_size == 0:
            txn.commit()
            print(f"Committed {counter} entries.")
            txn = env.begin(write=True)

# Final commit if needed
txn.commit()
print(f"Done. Saved {counter} strain segments to LMDB.")
env.close()
