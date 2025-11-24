# This script generates synthetic gravitational wave strains and stores them in an LMDB database.
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import pylab
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd 
import seaborn as sns
import time
from datetime import datetime
from tqdm.auto import tqdm
import os
import lmdb


def define_parameters():
    ''' This function defines the parameters for the gravitational waveform generation.
        Change the parameters here as needed.
        
    '''
    approximant='SEOBNRv4'     # The approximant to use. This can be any of the lalsimulation approximants.
    min_mass = 5               # Minimum mass of the binary system in M0. Default is 5.
    max_mass = 100                # Maximum mass of the binary system in M0. Default is 100.
    step_mass = 1              # Step of mass in Mo. Default is 1.
    min_distance = 50          # Minimum distance of the binary system in Mpc. Default is 50.
    max_distance = 500         # Maximum distance of the binary system in Mpc. Default is 500.
    step_distance = 50         # Step of distance in Mpc. Default is 50.
    sample_rate = 8192         # Sample rate of the waveform. Default is 8192.
    delta_t = 1.0/sample_rate  # Time step of the waveform. Default is 1/sample_rate.
    f_low = 10                 # Lowest frequency of the waveform in Hz. Default is 10.
    detector = 'H1'            # Choose between H1, L1, V1 detectors. Put all for all detectors. Default is H1
    ra = 0                     # The Right Ascension of the source in radians. Default is 0.
    dec = 0                    # The declination of the source in radians. Default is 0.
    polarization = 0           # The polarization angle of the source in radians. Default is 0
    
    inclinations =  [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2] # Inclination of the source in radians. Choose between 0 and pi rads
    spin1z = [-1, -0.5, 0, 0.5, 1]  # Dimensionless Spin of the first black hole in z-axis. Choose between -1 and 1
    spin2z = [-1, -0.5, 0, 0.5, 1]  # Dimensionless Spin of the second black hole in z-axis. Choose between -1 and 1
    
    parameters = {
        'approximant': approximant,
        'sample_rate': sample_rate,
        'delta_t': delta_t,
        'f_low': f_low,
        'detector': detector,
        'min_mass': min_mass,
        'max_mass': max_mass,
        'step_mass': step_mass,
        'min_distance': min_distance,
        'max_distance': max_distance,
        'step_distance': step_distance,
        'inclinations': inclinations,
        'spin1z': spin1z,
        'spin2z': spin2z,
        'ra': ra,
        'dec': dec,
        'polarization': polarization
    }
    
    return parameters
   
    
def to_detector(hp, hc, detector='all', ra=0, dec=0, polarization=0, sample_rate=8192):
    ''' 
    This function makes signals oriented with respect to the detectors.
    detector: The detector to use. This can be any of H1, L1, V1 detectors. Default is 'all'. 
    hp: The plus polarization of the gravitational-wave signal.
    hc: The cross polarization of the gravitational-wave signal.
    ra: The right ascension of the source in radians. Default is 0.
    dec: The declination of the source in radians. Default is 0.
    polarization: The polarization angle of the source in radians. Default is 0.
    '''
    
    if detector.lower() == 'h1' or detector.lower() == 'all':
        det_h1 = Detector('H1')

        h1 = det_h1.project_wave(hp, hc, ra=0, dec=0, polarization=0)
        
        h1_times = h1.sample_times
        
        #change shape of h1 |---------------------------------------------|
        h1_times=h1.sample_times
        h1_end_time = h1.end_time
        window_h1=h1_end_time-1

        for i in range(len(h1_times)):
            if h1_times[i]>=window_h1:
                # print(i)
                break

        h1_times = h1.sample_times[i:]
        h1 = h1[i:]

        if len(h1) < sample_rate:
            h1.size = h1.resize(sample_rate)
            old_size=len(h1_times)

            h1_times.size = h1_times.resize(sample_rate)

            for i in range(old_size,sample_rate):
                h1_times[i]=h1_times[i-1]+1/sample_rate

        # print(f'H1      = {h1.shape}')
        # print(f'H1_time = {h1_times.shape}')
        if detector.lower() == 'h1':
            return h1, h1_times

    if detector.lower() == 'l1' or detector.lower() == 'all':
        det_l1 = Detector('L1')
        
        l1 = det_l1.project_wave(hp, hc, ra=0, dec=0, polarization=0)
        
        l1_times = l1.sample_times
        
        #change shape of l1 |---------------------------------------------|
        l1_times=l1.sample_times
        l1_end_time = l1.end_time
        window_l1=l1_end_time-1

        for i in range(len(l1_times)):
            if l1_times[i]>=window_l1:
                # print(i)
                break

        l1_times = l1.sample_times[i:]
        l1 = l1[i:]

        if len(l1) < sample_rate:
            l1.size = l1.resize(sample_rate)
            old_size=len(l1_times)

            l1_times.size = l1_times.resize(sample_rate)

            for i in range(old_size,sample_rate):
                l1_times[i]=l1_times[i-1]+1/sample_rate

        # # print(f'L1      = {l1.shape}')
        # # print(f'L1_time = {l1_times.shape}')
        if detector.lower() == 'l1':
            return l1, l1_times
    
    if detector.lower() == 'v1' or detector.lower() == 'all':
        det_v1 = Detector('V1')
        
        v1 = det_v1.project_wave(hp, hc, ra=0, dec=0, polarization=0)

        v1_times = v1.sample_times

        #change shape of v1 |---------------------------------------------|
        v1_times=v1.sample_times
        v1_end_time = v1.end_time
        window_v1=v1_end_time-1

        for i in range(len(v1_times)):
            if v1_times[i]>=window_v1:
                # print(i)
                break
        
        v1_times = v1.sample_times[i:]
        v1 = v1[i:]

        if len(v1) < sample_rate:
            v1.size = v1.resize(sample_rate)
            old_size=len(v1_times)

            v1_times.size = v1_times.resize(sample_rate)

            for i in range(old_size,sample_rate):
                v1_times[i]=v1_times[i-1]+1/sample_rate
            
        # # print(f'V1      = {v1.shape}')
        # # print(f'V1_time = {v1_times.shape}')
        
        if detector.lower() == 'v1':
            return v1, v1_times
    
    if detector.lower() == 'all':
        return h1, h1_times, l1, l1_times, v1, v1_times

def remove_lmdb(path):
    if os.path.exists(path):

        try:
            os.remove(path + "/lock.mdb")
            os.remove(path + "/data.mdb")
            os.rmdir(path)
            print(f"Removed database at {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")
    else:
        print(f"No database found at {path} to remove.")

def remove_csv(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"Removed CSV file at {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")
    else:
        print(f"No CSV file found at {path} to remove.")
        
def get_file_size_gb(path):
    if not os.path.exists(path):
        return 0.0
    size_bytes = os.path.getsize(path)
    size_gb = size_bytes / (1024**3)
    return size_gb

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

def read_last_index_from_log(log_path):
    if not os.path.exists(log_path):
        return 0
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "LAST_INDEX" in line:
            try:
                return int(line.strip().split("=")[-1])
            except:
                pass
    return 0

def write_log(log_path, message):
    with open(log_path, "a") as logf:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logf.write(f"[{timestamp}] {message}\n")
        
def get_last_lmdb_index(env):
    with env.begin() as txn:
        cursor = txn.cursor()
        if cursor.last():
            last_key = cursor.key()
            counter = last_key.decode().split('_')[0]
            try:
                return int(counter)
            except ValueError:
                return 0
        else:
            return 0


def driver(
    batch_size,
    grawa_lmdb_path, 
    csv_path,
    log_path="generation.log"
):
    

    grawa_env = lmdb.open(grawa_lmdb_path, map_size=int(1e12))  # Adjust map_size as needed

    print("Starting waveform generation...")
    start_index = max(read_last_index_from_log(log_path), get_last_lmdb_index(grawa_env))
    
    write_log(log_path, f"Starting generation from index {start_index}")
    
    if start_index != 0:
        print('Searching existing keys in LMDB database.')
        with grawa_env.begin(write=False) as grawa_txn:
            lmdb_keys = [key.decode() for key, _ in grawa_txn.cursor()]
            print('Keys count:', len(lmdb_keys))
    else: 
        lmdb_keys = []
            
    
    csv_exists = os.path.exists(csv_path)
    csvfile = open(csv_path, mode='a', newline="") if csv_exists else open(csv_path, mode='w', newline="")
    
    fieldnames = ["counter", "m1", "m2", "distance", "inclination", "spin1z", "spin2z"]
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if not csv_exists:
        csv_writer.writeheader()

    params = define_parameters()
    approximant = params['approximant']
    sample_rate = params['sample_rate']
    delta_t = params['delta_t']
    f_low = params['f_low']
    detector = params['detector']
    min_mass = params['min_mass']
    max_mass = params['max_mass']
    step_mass = params['step_mass']
    min_distance = params['min_distance']
    max_distance = params['max_distance']
    step_distance = params['step_distance']
    inclinations = params['inclinations']
    spin1z = params['spin1z']
    spin2z = params['spin2z']
    ra = params['ra']
    dec = params['dec']
    polarization = params['polarization']
    
    csv_batch_buffer = []
    strains = []
    keys = []

    csv_size = 0
    db_size = 0
    counter=0
    
    starting_time=time.time()        
    for distance in range(min_distance, max_distance+1, step_distance):

        with tqdm(range(min_mass, max_mass+1, step_mass)) as t:
            for i in t:
                m2=i
                j=0
                # print(f'Mass2 = {m2}')      #Mo
                while m2+j<=max_mass:
                    m1=m2+j
                    j+=1
                    
                    # print(f'Mass1 = {m1}')

                    for inclination in inclinations:
                        for spin1 in spin1z:
                            for spin2 in spin2z:
                                counter+=1
                                
                                row_id = f"{counter:09}_{m1}_{m2}_{distance}_{np.round(inclination*180/np.pi)}_{spin1}_{spin2}"
                                
                                if row_id not in lmdb_keys:
                                    hp, hc = get_td_waveform(approximant=approximant,
                                                            mass1=m1,
                                                            mass2=m2,
                                                            spin1z=spin1,
                                                            spin2z=spin2,
                                                            inclination=inclination,
                                                            delta_t= delta_t,
                                                            distance=distance,
                                                            f_lower=f_low)
                                    
                                    h1 , _ = to_detector(hp, hc, detector, ra, dec, polarization, sample_rate)
                                    
                                    #set tqdm bars
                                    t.set_description(f'Distance:{distance}/{max_distance} | M1:{m1}/{max_mass} | M2:{m2}/{max_mass}')
                                    t.set_postfix(csv= f'{csv_size:.5f} GB', db= f'{db_size:.5f} GB', time=f'{time.time()-starting_time:.3f} sec')                   

                                    
                                    csv_batch_buffer.append({'counter': f'{counter:09}', 'm1': m1, 'm2': m2, 'distance': distance, 'inclination': np.round(inclination*180/np.pi), 'spin1z': spin1, 'spin2z': spin2})
                                    
                                    strains.append(h1.numpy().astype(np.float64))
                                    keys.append(row_id)
                                    
                                    if counter % batch_size == 0:
                                        with grawa_env.begin(write=True) as grawa_txn:
                                            for i in range(batch_size):
                                                grawa_txn.put(keys[i].encode(), strains[i].tobytes())

                                        csv_writer.writerows(csv_batch_buffer)
                                        csvfile.flush()
                                        strains = []
                                        keys = []
                                        csv_batch_buffer = []
                                        
                                        write_log(log_path, f'LAST_INDEX={counter}')
                                        write_log(log_path, f'Samples {counter-batch_size} to {counter} written successfully.')
                                        db_size = get_lmdb_size_gb(grawa_lmdb_path)
                                        csv_size = get_file_size_gb(csv_path)
    
    with grawa_env.begin(write=True) as grawa_txn:
        for i in range(len(strains)):
            grawa_txn.put(keys[i].encode(), strains[i].tobytes())

    csv_writer.writerows(csv_batch_buffer)
    csvfile.flush()
    strains = []
    keys = []
    csv_batch_buffer = []
    
    write_log(log_path, f'LAST_INDEX={counter}')
    write_log(log_path, f'Samples {counter-batch_size+1} to {counter} written successfully.')
    db_size = get_lmdb_size_gb(grawa_lmdb_path)
    csv_size = get_file_size_gb(csv_path)
                                                          
    ending_time=time.time()
    tqdm.write(f'\nData Generating Time = {ending_time-starting_time:.3f} seconds')
    tqdm.write(f'\nTotal Data = {counter} samples') 
    tqdm.write(f'LMDB size: {db_size} GB')       
    write_log(log_path, f"Generation completed up to index {counter}.")
    csvfile.close()

       
def main():
    remove_database = False    # Set to True to remove existing databases and CSV file before running
    continue_from_last = True  # Set to True to continue from the last index in the log file
    batch_size = 10000         # Number of gravitational-wave strain samples stored in the LMDB at once
    
    path = '/home/sakellariou/hero_disk/test/'   # <---- change here to the path you want to save the lmdb
    grawa_lmdb_path = path + "grawa.lmdb"   
    csv_path = path + "grawa.csv"
    log_path = path + "generation.log"
    
    if remove_database:
        print("Removing existing databases and CSV file...")
        remove_lmdb(grawa_lmdb_path)
        remove_csv(csv_path)
        
        if os.path.exists(log_path):
                    try:
                        os.remove(log_path)
                        print(f"Removed log file at {log_path}")
                    except Exception as e:
                        print(f"Error removing log file: {e}")

        print("Existing databases and CSV file removed.")
        
    if not continue_from_last:
        if os.path.exists(grawa_lmdb_path):
            raise FileExistsError(f"LMDB database {grawa_lmdb_path} already exists. Please remove it before running.")
        if os.path.exists(csv_path):
            raise FileExistsError(f"CSV file {csv_path} already exists. Please remove it before running.")
    
    driver(batch_size, grawa_lmdb_path, csv_path, log_path) 
    
if __name__ == '__main__':
    main()