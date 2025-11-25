# hybrid-bbh-parameter-estimation
Code accompanying the paper "Binary Black Hole Parameter Estimation with Hybrid CNN–Transformer Neural Networks", published in [Astronomy and Computing](https://doi.org/10.1016/j.ascom.2025.101027).

## 1. Data Generation

In the initial phase, a dataset of raw, noiseless gravitational wave strains is generated. Due to the substantial volume of the data (~380 GB for the default configuration), LMDB (Lightning Memory-Mapped Database) is utilized. This approach ensures memory-efficient storage and rapid access, eliminating the need to load the entire dataset into RAM.

To generate your own database, use the `data_generator.py` script.

### Configuration
Before running the script, open `data_generator.py` and modify the following:

1.  **Output Paths:** Set the directory paths where you want the LMDB database, the CSV index, and the log file to be stored.
2.  **Parameters:** You can customize the physical parameters of the generated waveforms (e.g., mass ranges, spins) by modifying the `define_parameters()` function inside the script.

### Data Structure
* **LMDB Keys:** The keys in the database are constructed directly from the physical parameters used to generate each specific waveform. This ensures a strict 1-to-1 mapping between the physics and the stored data.
* **CSV Index:** A companion CSV file is generated to allow for human-readable inspection of the dataset parameters without opening the binary database.
  
### Running the Generator
```bash
python data_generator.py
```

### Outputs
The script will produce three files:
* **LMDB Database:** A binary database containing the raw waveform strains.
* **CSV File:** A metadata index for human readability (contains labels and parameters).
* **Log File:** Records the generation process and any errors.

**⚠️ Storage Warning:** The default configuration generates a dataset of approximately **380 GB**. Ensure you have sufficient disk space before running this script.

## 2. Data Preprocessing and Dataset Construction

Following the generation of raw strains, the data undergoes preprocessing, noise injection, and normalization before being converted into optimized TensorFlow Datasets. This process is handled by the `final_datasets.py` script.

### The Pipeline
The script executes the following procedure:

1.  **Splitting and Shuffling:**
    The keys from the initial LMDB are shuffled and partitioned into **Training**, **Validation**, and **Testing** sets. To ensure consistency across future runs, these keys are preserved in separate NumPy (`.npy`) files.

2.  **Noise Injection and Intermediate Storage:**
    * **Training Set:** Raw strains are retrieved using the training keys and combined with Gaussian noise. These noisy strains are temporarily stored in an intermediate LMDB database to facilitate global statistic computation.
    * **Validation/Test Sets:** Raw strains are similarly retrieved and combined with Gaussian noise.

3.  **Normalization (Min-Max):**
    * Global minimum and maximum values are extracted from the **Training** set (via the intermediate LMDB) and stored in `.npy` files.
    * **Min-Max Normalization** is applied to the Training set using these values.
    * Crucially, the Validation and Testing sets are normalized using the **same** minimum and maximum values derived from the Training set. This ensures no information regarding the range of the test data is leaked to the model.

4.  **Final Format Conversion:**
    The normalized strains and their corresponding parameters are serialized into **TensorFlow Datasets**. This format is selected for its superior I/O throughput during training and memory efficiency compared to LMDB.

### Running the Script
To execute this process, configure the input (initial LMDB) and output paths in `final_datasets.py`, then run:

```bash
python final_datasets.py
```

### Outputs
The execution results in the following files:
* **Key Files:** `train_keys.npy`, `val_keys.npy`, `test_keys.npy`.
* **Normalization Statistics:** `minimum.npy`, `maximum.npy`.
* **Final Datasets:** Three TensorFlow Datasets (Training, Validation, Testing).
* **Intermediate Artifacts:** A temporary LMDB database and log file (these may be deleted after successful execution).

### Download Ready-to-Use Datasets
If you wish to skip the generation and preprocessing steps, the final processed TensorFlow Datasets are available for download here:
* [Training Dataset](LINK_HERE)
* [Validation Dataset](LINK_HERE)
* [Testing Dataset](LINK_HERE)

## 3. Model Definition, Training and Evaluation

The core model architecture is implemented in **PyTorch**. To interface efficiently with the pre-generated TensorFlow Datasets, a custom data ingestion pipeline is employed.

### Hardware Requirements
**GPU acceleration for PyTorch is strictly recommended.** Due to the complexity of the hybrid CNN-Transformer architecture and the size of the datasets, training on a CPU is computationally prohibitive.
* **Benchmark:** For reference, full convergence requires approximately **1.5 weeks** on a single **NVIDIA A100 GPU**.

### Data Ingestion (TF-to-PyTorch Bridge)
A custom class inheriting from PyTorch's `IterableDataset` is utilized to stream data directly from the TensorFlow Datasets.
* **Efficiency:** This approach prevents reshuffling overhead and eliminates the need to load the full dataset into RAM.
* **Batching:** Data is converted to PyTorch tensors on-the-fly and batched using the standard PyTorch `DataLoader`.

### Model Configuration
Distinct classes are provided for each model variant presented in the paper. The models are configured to regress **6 physical parameters**.
* **Loss Function:** L1 Loss (Mean Absolute Error).
* **Optimizer:** Adam.
* **Early Stopping:** Support for early stopping is implemented (disabled by default) with configurable patience.

### Training Protocol
The training process is managed by a dedicated function that accepts the number of epochs and checkpoint preferences.
* **Checkpointing:** To ensure training stability, a checkpoint is saved at every epoch. This includes the model state, optimizer state, epoch number, and loss history, allowing training to resume seamlessly in case of interruption.
* **Best Model Preservation:** The model weights corresponding to the minimum validation loss are automatically tracked and saved to a `.pth` file.
* **Visualization:** Train and Validation loss curves are returned and plotted to visualize convergence.

### Evaluation
Upon completion of training, the "Best Model" weights are loaded for evaluation.
* **Metrics:** Mean Absolute Error (MAE) and other error metrics are computed and serialized to a `.json` file for readability.
* **Visual Inspection:** A function is provided to plot real vs. predicted values for a subset of waveforms, allowing for qualitative assessment of model performance.

### Running the Training
To initiate training, configure the dataset paths and hyperparameters in `training_evaluation.py`, then run:

```bash
python train_network.py
```

### Outputs
Upon completion, a results directory is generated containing:
* `best_model.pth`: The model weights yielding the lowest validation loss.
* `checkpoint.pth`: The final state of the model and optimizer.
* `loss.png`: A figure visualizing the Training vs. Validation loss over epochs.
* `metrics.json`: A generic file containing Mean Absolute Error (MAE) and other evaluation metrics.



