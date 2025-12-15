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

**⚠️ Runtime Warning:** The default configuration requires a significant amount of time to complete. Expect the process to take several days, potentially up to 2–3 weeks. Ensure your system can remain active and uninterrupted for this duration.

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

### ⬇️ Download Ready-to-Use Datasets

If you wish to skip the data generation and preprocessing steps, you can download the final processed TensorFlow Datasets directly.

**File Size:** ~321 GB

* **Direct Download:** [Download Datasets (.zip)](http://sgw.plagianakos.gr:8081/gw_datasets.zip)

**Command Line (for remote servers / Colab):**
```bash
# Download the dataset
wget [http://sgw.plagianakos.gr:8081/gw_datasets.zip](http://sgw.plagianakos.gr:8081/gw_datasets.zip)

# Unzip the files
unzip gw_datasets.zip
```

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

## 4. Comparative Analysis and Visualization

To assess the performance improvements offered by the hybrid architecture, the `parameter_analysis.ipynb` Jupyter Notebook is provided. This tool facilitates a direct comparison between the baseline CNN model and the Hybrid CNN-Transformer.

### Functionality
The notebook performs an evaluation of both the standard CNN and the Hybrid models. The primary output consists of **scatter plots** illustrating the correlation between true injected parameters and the model-predicted values across the testing set.

### Configuration
Prior to execution, the following parameters must be defined within the notebook:
1.  **Dataset Path:** The location of the pre-processed TensorFlow Test Dataset.
2.  **Model Selection:** The specific architecture to be evaluated (Standard CNN or Hybrid).
3.  **Weights Path:** The directory containing the `best_model.pth` file generated during the training phase.
4.  **Output Directory:** The destination folder where the generated figures will be saved.

### Execution
```bash
jupyter notebook parameter_analysis.ipynb
```

## 5. Latent Space Analysis (t-SNE)

To investigate the internal representations learned by the model, the `plot_features.ipynb` notebook is utilized. This tool employs **t-Distributed Stochastic Neighbor Embedding (t-SNE)** to project the high-dimensional latent space into 2D visualizations.

### Functionality
The notebook extracts feature maps from a specified layer of the neural network and generates:
* **Individual Parameter Plots:** t-SNE projections colored by the value of each physical parameter.
* **Composite Mass Plot:** A combined visualization specifically analyzing the structuring of the mass parameters ($m_1$ and $m_2$) within the latent space.

### Configuration
The following parameters must be configured within the notebook:
1.  **Dataset Path:** The directory containing the pre-processed TensorFlow Datasets.
2.  **Model Selection:** The specific architecture to be analyzed (e.g., Standard CNN or Hybrid).
3.  **Weights Path:** The location of the best-performing model weights (`.pth` file).
4.  **Target Layer:** The index of the specific neural network layer from which the latent space representations are to be extracted.
5.  **Output Destination:** The directory where the resulting t-SNE plots will be stored.
   
### Execution
```bash
jupyter notebook plot_features.ipynb
```

## 6. Real Gravitational-Wave Data Acquisition

To evaluate the model on empirical observational data, the `real_gw_to_lmdb.py` script is employed. This tool interfaces with the **Gravitational Wave Open Science Center (GWOSC)** to automatically retrieve and format events from the H1 detector.

### Processing Pipeline
The script iterates through all available observing runs and performs the following operations:

1.  **Data Retrieval:** Queries GWOSC for 32-second strain segments at a native sampling rate of 16384 Hz.
2.  **Integrity Check:** Scans for missing data (`NaN` values); corrupted segments are automatically excluded.
3.  **Resampling:** Downsamples the strain to **8192 Hz** to match the input specifications of the trained neural networks.
4.  **Cropping:** Extracts a precise **1-second window** centered around the event time.
5.  **Serialization:** The processed waveforms are stored in an **LMDB database**. This format is selected to maintain compatibility with the existing data ingestion pipelines used during the training phase.

### Configuration
Before running the script, open `real_gw_to_lmdb.py` and modify the following:

* **Output Paths:** Set the directory paths where you want the LMDB database, and the CSV file to be stored.

### Outputs
The execution results in two primary files:
* **LMDB Database:** Contains the processed real waveforms. The retrieval keys are constructed based on the event's source parameters (masses and distance).
* **CSV Index:** A metadata file containing the lookup keys, official **Event Names**, and the reference parameter values (Masses, Distance) along with their **Upper and Lower Bounds**.

### Execution
```bash
python real_gw_to_lmdb.py
```

## 7. Real Data Filtering and Fine-Tuning Preparation

To prepare the observational data for domain adaptation (fine-tuning), the `real_gw_tf_datasets.py` script is employed. This tool filters the real event database and constructs TensorFlow Datasets specifically designed for cross-validation experiments.

Note: This script requires lmdb_dataset_handler.py to be present in the same directory, as it depends on helper classes defined therein.

### Filtering Logic
The script scans the LMDB of real events and retains only those whose estimated parameters (masses and distance) fall within the specific training range of the pre-trained Neural Networks.
* **Result:** Under the default configuration, **5 confirmed events** match these criteria and are selected for analysis.

### Dataset Construction (Cross-Validation Slices)
The script generates specific TensorFlow Datasets to support two distinct fine-tuning strategies. For every dataset slice, **Min-Max normalization** is applied using statistics derived strictly from the *training* portion of that specific fold to prevent data leakage.

#### Strategy A: Single-Event Training (1-vs-4)
* **Objective:** To test the model's ability to learn from a single real-world example.
* **Protocol:** The model is fine-tuned on **1 event** and evaluated on the remaining **4 events**.
* **Output:** 5 distinct dataset splits.
* **Naming Convention:** Datasets are identified by the pattern `slice[N]` (e.g., `slice1_train_dataset`).

#### Strategy B: Leave-One-Out Training (4-vs-1)
* **Objective:** To test the model's generalization by maximizing the real-world training data.
* **Protocol:** The model is fine-tuned on **4 events** and evaluated on the remaining **1 event**.
* **Output:** 5 distinct dataset splits.
* **Naming Convention:** Datasets are identified by the pattern `slice[N]_reverse` (e.g., `slice1_reverse_train_dataset`).

### Configuration
Before executing the script, the following paths must be defined within `real_gw_tf_datasets.py`:
1.  **Source LMDB:** Path to the database containing the processed real gravitational waves.
2.  **Source CSV:** Path to the metadata file generated in the previous step.
3.  **Destination Directory:** The folder where the resulting TensorFlow Datasets will be saved.

### Execution
```bash
python real_gw_tf_datasets.py
```

## 8. Fine-Tuning and Statistical Evaluation

Following the dataset preparation, the model undergoes domain adaptation (fine-tuning) using the `real_gw_slice_policy.py` script, followed by statistical aggregation using `real_gw_slice_policy.ipynb`.

### Phase A: Fine-Tuning Execution
The `real_gw_slice_policy.py` script performs training and evaluation for both the **DeepModel** (Standard CNN) and its **Hybrid** variant. The process iterates through every dataset slice (both Strategy A and Strategy B).

#### Methodology
1.  **Weight Initialization:** For each slice, the model is initialized with the best weights obtained from the training on artificial strains (Step 3).
2.  **Training:** The model is fine-tuned on the specific training slice.
3.  **Boundary Evaluation:** A specialized evaluation metric is computed using the Upper and Lower bounds provided in the source CSV.
    * If a prediction falls *within* the GWOSC confidence bounds: **MAE = 0**.
    * Otherwise: Standard MAE is calculated.

#### Configuration
Before execution, update the following paths in `real_gw_slice_policy.py`:
1.  **Datasets & CSV:** Paths to the "sliced" TensorFlow Datasets and the real event metadata CSV.
2.  **Pre-trained Weights:** Path to the `best_model.pth` from the simulation training phase.
3.  **Output Destination:** The root directory for results. The script automatically generates a hierarchical structure: `Main_Folder` -> `Model_Name_Slice_Name_Folder`.

#### Outputs
* **Results CSV:** A consolidated file containing metrics for all slices.
* **Fine-Tuned Weights:** Saved specifically for each slice/model combination.
* **Loss Figures:** Visualization of the fine-tuning convergence.

```bash
python real_gw_slice_policy.py
```

### Phase B: Statistical Analysis and Supplementary Data
The `real_gw_slice_policy.ipynb` notebook is utilized to aggregate the results and generate the data presented in the paper's **Supplementary Table**.

#### Functionality
This notebook loads the results CSV and the fine-tuned weights to compute:
* **Standard Metrics:** Mean MAE ($\pm$ Std) for standard slices and reverse slices.
* **Boundary Metrics:** Mean Boundary MAE ($\pm$ Std) for standard and reverse slices.
* **Verification:** Loads the fine-tuned weights to display a direct comparison of True vs. Predicted values for manual inspection.

#### Configuration
Required inputs for the notebook:
1.  **Data Sources:** Paths to the Datasets and CSV.
2.  **Model Weights:** Path to the directory containing the *fine-tuned* weights (generated in Phase A).
3.  **Results File:** Path to the consolidated CSV generated in Phase A.

```bash
jupyter notebook real_gw_slice_policy.ipynb
```

## 9. Posterior Estimation with Normalizing Flows

The final stage of the pipeline transitions from point-parameter estimation to full posterior density estimation. This is achieved using **Normalizing Flows** (via the `nflows` library) conditioned on the embeddings from the main neural network.

### Phase A: Feature Normalization (Prerequisite)
Before training the flow, the statistical properties (Mean and Standard Deviation) of the training dataset parameters must be extracted to normalize the target distribution.

#### Execution
Run the `mean_std_extractor.py` script.
* **Input:** Path to the training dataset (artificial strains).
* **Output:** Generates `parameters_mean.npy` and `parameters_std.npy` in the same directory.

```bash
python mean_std_extractor.py
```

### Phase B: Flow Training and Evaluation
The `prob_training_evaluation.py` script integrates the Normalizing Flow (`nflows`) with the pre-trained CNN/Hybrid model to estimate full posterior distributions.

#### Methodology
* **Loss Function:** The model minimizes the **Negative Log-Likelihood (NLL)**.
* **Freezing Policy:** By default, the main feature extractor (CNN/Transformer) is initialized with the best pre-trained weights (Step 3) and **frozen**. Only the Normalizing Flow layers are trained to learn the probability density.
    * *Note:* This behavior is configurable; the backbone model can be unfrozen for end-to-end fine-tuning if desired.

#### Advanced Metrics & Visualization
In addition to standard error metrics, this stage computes probabilistic diagnostics:
* **CRPS (Continuous Ranked Probability Score):** Measures the accuracy of the predicted cumulative distribution.
* **Sharpness:** Quantifies the concentration (width) of the posterior distributions.
* **Combined p-value:** A statistical test of consistency between the predicted and true distributions.
* **P-P Plot:** A Probability-Probability plot is generated to visually assess calibration (should lie on the diagonal).

#### Configuration
Update the following paths in `prob_training_evaluation.py`:
1.  **Statistics:** Paths to `parameters_mean.npy` and `parameters_std.npy`.
2.  **Dataset Path:** The directory containing the TensorFlow Datasets (Training/Validation/Testing).
3.  **Pre-trained Weights:** Path to the `best_model.pth` (from the initial simulation training) to initialize the frozen backbone.
4.  **Destination Folder:** The directory where the resulting models, plots, and metrics will be saved.
5.  **Hyperparameters:** Batch size, Learning Rate (LR), and Optimizer settings.

#### Execution
```bash
python prob_training_evaluation.py
```

### Outputs
Upon completion, a results directory is generated containing:
* **`best_model.pth`**: The weights of the Normalizing Flow model yielding the lowest Negative Log-Likelihood (NLL).
* **`checkpoint.pth`**: The final state of the model and optimizer (useful for resuming training).
* **`loss.png`**: A figure visualizing the NLL loss convergence over epochs.
* **`pp_plot.png`**: The Probability-Probability (P-P) plot, used to visually verify that the estimated posteriors are well-calibrated.
* **`metrics.json`**: An enhanced metrics file containing standard errors plus probabilistic scores:
    * **CRPS** (Continuous Ranked Probability Score)
    * **Sharpness**
    * **Combined p-value**

## Citation
If you use this code or datasets in your research, please cite our paper:

**BibTeX:**
```bibtex
@article{SAKELLARIOU2026101027,
title = {Binary black hole parameter estimation with hybrid CNN-Transformer Neural Networks},
journal = {Astronomy and Computing},
volume = {54},
pages = {101027},
year = {2026},
issn = {2213-1337},
doi = {https://doi.org/10.1016/j.ascom.2025.101027},
url = {https://www.sciencedirect.com/science/article/pii/S2213133725001003},
author = {Panagiotis N. Sakellariou and Spiros V. Georgakopoulos and Sotiris Tasoulis and Vassilis P. Plagianakos},
}
```
