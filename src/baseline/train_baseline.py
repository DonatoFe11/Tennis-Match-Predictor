import pandas as pd
import numpy as np
import time
import warnings
import torch
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import log_loss
from tqdm import tqdm
import re
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our custom modules
from src.data_processing.data_loader import load_and_prepare_data
from src.baseline.feature_engineering import engineer_historical_features

def prepare_modeling_data(enriched_df):
    """
    Takes the feature-enriched DataFrame and prepares the final X and y matrices for modeling.
    """
    print("\n--- Structuring Final DataFrame for Modeling ---")

    # --- 1. Casually swap winner and loser roles for half of the matches ---
    # To ensure that the model can learn from both perspectives, we randomly swap winner and loser roles for half of the matches.
    np.random.seed(42)
    swap_mask = np.random.rand(len(enriched_df)) < 0.5
    
    df1 = enriched_df.copy() # Here the winner is p1
    df2 = enriched_df.copy() # Here the loser is p1

    # Define all columns to be renamed for p1 and p2
    p_cols = ['id', 'name', 'elo', 'rank', 'age', 'h2h', 'fatigue', 'streak']
    winner_map = {f'winner_{c}': f'p1_{c}' for c in p_cols}
    loser_map = {f'loser_{c}': f'p2_{c}' for c in p_cols}
    
    df1 = df1.rename(columns={**winner_map, **loser_map})
    df1['p1_wins'] = 1
    
    # Correctly swap winner/loser roles for df2
    swapped_winner_map = {f'winner_{c}': f'p2_{c}' for c in p_cols}
    swapped_loser_map = {f'loser_{c}': f'p1_{c}' for c in p_cols}
    df2 = df2.rename(columns={**swapped_winner_map, **swapped_loser_map})
    df2['p1_wins'] = 0

    model_df = pd.concat([df1[~swap_mask], df2[swap_mask]]).sort_index()

    # --- 2. Engineer differential features ---
    # Calculate the difference between p1 and p2 for each feature 
    for col in ['elo', 'rank', 'h2h', 'fatigue', 'streak', 'age']:
        if f'p1_{col}' in model_df.columns and f'p2_{col}' in model_df.columns:
            model_df[f'{col}_diff'] = model_df[f'p1_{col}'] - model_df[f'p2_{col}']

    # --- 3. Handling categorical features ---
    surface_dummies = pd.get_dummies(model_df['surface'], prefix='surface', dummy_na=False)
    model_df = pd.concat([model_df, surface_dummies], axis=1)

    # Programmatically define features - selecting columns that contain '_diff' or 'surface_'
    features = [col for col in model_df.columns if '_diff' in col]
    features.extend([col for col in model_df.columns if 'surface_' in col])
    target = 'p1_wins'

    # Drop rows with NaNs in crucial columns
    final_model_df = model_df.dropna(subset=['rank_diff', 'age_diff'])
    
    # Create X as to only include the engineered features
    X = final_model_df[features].copy() 
    y = final_model_df[target].copy()

    # Sanitize column names - delete any special characters that might cause issues
    X.columns = [re.sub(r"\[|\]|<", "_", col) for col in X.columns]
    
    print("Final X and y matrices created.")
    return X, y

def run_evaluation():
    warnings.filterwarnings('ignore', category=UserWarning)

    # --- Step 1 & 2: Load and Engineer Features ---
    raw_df = load_and_prepare_data()
    enriched_df = engineer_historical_features(raw_df)
    
    # --- Step 3: Prepare Data for Modeling ---
    X, y = prepare_modeling_data(enriched_df)

    # --- Step 4: Evaluation ---
    print("\n--- Section 4: Starting Evaluation with JSON Logging ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    param_dist = {
        'learning_rate': uniform(0.01, 0.15), 'max_depth': randint(3, 10),
        'n_estimators': randint(500, 2500), 'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3), 'gamma': uniform(0, 0.5)
    }
    N_RUNS = 10
    
    results_filepath = "results/baseline_results.json"
    all_run_results = []
    print("\nStarting a new evaluation session. Previous results file will be overwritten.")

    # --- Main Loop for 10 Independent Runs ---
    for i in tqdm(range(N_RUNS), desc="Overall Baseline Runs"):
        run_start_time = time.time()
        
        # Ensure that traning data is always before the test data
        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds, oof_true = [], []
        best_params_per_fold = []

        for fold, (train_index, test_index) in tqdm(enumerate(tscv.split(X)), total=tscv.n_splits, desc=f"Run {i+1} Folds", leave=False):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            
            xgb_clf = xgb.XGBClassifier(device=device, tree_method='hist', objective='binary:logistic', eval_metric='logloss')
            
            random_search = RandomizedSearchCV(
                estimator=xgb_clf, param_distributions=param_dist, n_iter=5, 
                cv=3, scoring='neg_log_loss', random_state=i*10 + fold, n_jobs=1, verbose=0
            )
            
            random_search.fit(X_train_fold, y_train_fold)
            best_model = random_search.best_estimator_
            best_params_per_fold.append(random_search.best_params_)

            y_pred_proba = best_model.predict_proba(X_test_fold)[:, 1]
            oof_preds.extend(y_pred_proba)
            oof_true.extend(y_test_fold)

        run_logloss = log_loss(oof_true, oof_preds)
        run_duration = time.time() - run_start_time
        
        run_summary = {
            'run_id': i + 1,
            'log_loss': run_logloss,
            'duration_seconds': run_duration,
            'best_params_per_fold': best_params_per_fold 
        }
        all_run_results.append(run_summary)
        
        with open(results_filepath, 'w') as f:
            json.dump(all_run_results, f, indent=4)

        tqdm.write(f"-> Run {i+1} complete in {run_duration:.2f}s. Overall Log-Loss: {run_logloss:.4f}")

    # --- Final Aggregated Results ---
    results_df = pd.DataFrame(all_run_results)
    baseline_scores_for_ttest = results_df['log_loss']

    print("\n--- Baseline Model Final Performance Summary ---")
    print(f"Log-Loss over {N_RUNS} runs:\n{baseline_scores_for_ttest.describe().round(4)}")


if __name__ == "__main__":
    run_evaluation()