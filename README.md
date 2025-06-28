# Tennis Match Prediction with Graph Neural Networks

## Overview

This repository contains the complete implementation for a Master's level project in Machine Learning, focusing on predicting the outcomes of professional men's (ATP) tennis matches. The core of the project is a comparative analysis between a traditional, feature-rich **XGBoost** model and a modern **Graph Neural Network (GNN)**.

The GNN approach models the entire ecosystem of players as a relational graph, learning rich player embeddings to capture nuanced patterns of skill and historical context. The project demonstrates that this graph-based model significantly and statistically outperforms the strong baseline.

## Features

- **Robust Baseline Model**: An XGBoost classifier trained on a rich set of engineered features, including Elo ratings, Head-to-Head stats, player fatigue, and winning streaks.
- **Advanced GNN Model**: A GraphSAGE-based network that learns player embeddings from scratch, evaluated using a rigorous temporal cross-validation protocol.
- **Modular Code Structure**: The code is organized into distinct modules for data loading, feature engineering, and model training for clarity and reusability.
- **In-depth Analysis Notebook**: A final Jupyter Notebook (`tennis_match_prediction_with_GNN.ipynb`) that presents the results, statistical analysis (t-test), and qualitative insights derived from the models (t-SNE visualization).

---

## Project Structure

```
TENNIS_MATCH_PREDICTOR/
├── data/
│   └── tennis_atp/
│       └── ... (matches .csv files)
│
├── notebooks/
│   └── tennis_match_prediction_with_GNN.ipynb
│
├── results/
│   ├── baseline_results.json
│   ├── gnn_results.json
│   ├── best_gnn_embeddings.pt
│   └── best_player_map.json
│
├── src/
│   ├── __init__.py
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   └── train_baseline.py
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   │
│   └── gnn/
│       ├── __init__.py
│       ├── graph_builder.py
│       └── train_gnn.py
│
├── .gitignore
├── environment.yml
└── README.md
```

---

## Setup and Installation

### 1. Clone the Repository

```sh
git clone [URL_DEL_TUO_REPOSITORY_GITHUB]
cd TENNIS_MATCH_PREDICTOR
```
*(Remember to replace the URL with your actual GitHub repository URL)*

### 2. Download the Dataset

The project uses the publicly available dataset from Jeff Sackmann. Please clone it into the `data/` directory:

```sh
# Navigate to the data directory
cd data

# Clone the repository
git clone https://github.com/JeffSackmann/tennis_atp.git

# Return to the project root
cd ..
```

### 3. Create the Conda Environment

This project uses a Conda environment to manage its dependencies. To create and activate it, run the following commands:

```bash
# Create the environment from the .yml file
conda env create -f environment.yml

# Activate the new environment
conda activate tennis-ml
```
This will install all necessary packages, including PyTorch with CUDA support and PyTorch Geometric.

---

## Running the Experiments

The experiments are designed to be run as standalone Python scripts from the **root directory of the project**.

### 1. Run the XGBoost Baseline Evaluation

This script will execute the entire baseline pipeline, including feature engineering and the 10-run temporal cross-validation. Results will be saved to `results/baseline_results.json`.

```sh
python -m src.baseline.train_baseline
```

### 2. Run the GNN Model Evaluation

This script will run the full GNN pipeline, including graph construction and the 10-run evaluation with varied hyperparameters. Results will be saved to `results/gnn_results.json`.

```sh
python -m src.gnn.train_gnn
```
**Note:** Both scripts are computationally intensive and may take a significant amount of time to complete, depending on your hardware.

---

## Viewing the Results and Analysis

After running the experiments, all results and in-depth analysis can be found in the main Jupyter Notebook.

1.  **Launch Jupyter Lab:**
    ```sh
    jupyter lab
    ```
2.  **Open the Notebook:**
    Navigate to the `notebooks/` directory and open `tennis_match_prediction_with_GNN.ipynb`.

The notebook is structured as a final report, loading the `.json` files from the `results/` directory to display performance comparison tables, statistical tests, and qualitative visualizations like the t-SNE plot of player embeddings.