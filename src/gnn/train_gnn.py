# src/gnn/train_gnn.py (Final Version with Robust Evaluation and Embedding Saving)

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# --- Add Project Root to Python Path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_processing.data_loader import load_and_prepare_data
from src.baseline.feature_engineering import engineer_historical_features

# ==============================================================================
# SECTION 1: GNN MODEL AND HELPER FUNCTIONS
# ==============================================================================

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = SAGEConv(embedding_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def encode(self, edge_index):
        x = self.embedding.weight
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        source_embeds = z[edge_label_index[0]]
        dest_embeds = z[edge_label_index[1]]
        return (source_embeds * dest_embeds).sum(dim=-1)

def train_one_epoch(model, train_graph, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    
    z = model.encode(train_graph.edge_index)
    
    neg_edge_index = negative_sampling(
        edge_index=train_graph.edge_index, num_nodes=train_graph.num_nodes,
        num_neg_samples=train_graph.edge_index.size(1)
    ).to(device)
    
    edge_label_index = torch.cat([train_graph.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(train_graph.edge_index.size(1)), 
        torch.zeros(neg_edge_index.size(1))
    ], dim=0).to(device)
    
    out = model.decode(z, edge_label_index)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, train_graph, test_idx, test_labels):
    model.eval()
    z = model.encode(train_graph.edge_index)
    out = model.decode(z, test_idx)
    probs = out.sigmoid().cpu().numpy()
    true_labels = test_labels.cpu().numpy()
    
    try:
        auc = roc_auc_score(true_labels, probs)
        loss = log_loss(true_labels, probs)
    except ValueError:
        auc, loss = 0.5, -1.0 # Default values if only one class is present
    
    return auc, loss, probs

# ==============================================================================
# SECTION 2: MAIN EVALUATION PIPELINE
# ==============================================================================

def run_gnn_evaluation_pipeline():
    # --- Step 1 & 2: Load Data and Engineer Features ---
    raw_df = load_and_prepare_data(path_pattern='data/tennis_atp/atp_matches_*.csv')
    enriched_df = engineer_historical_features(raw_df)

    # --- Step 3: Setup for Multiple Runs & Hyperparameter Space ---
    N_RUNS = 10
    param_grid = [
        {'lr': 0.005, 'embedding_dim': 128, 'hidden_channels': 128, 'out_channels': 64, 'dropout': 0.5, 'epochs': 200},
        {'lr': 0.01,  'embedding_dim': 128, 'hidden_channels': 256, 'out_channels': 128,'dropout': 0.4, 'epochs': 150},
        {'lr': 0.001, 'embedding_dim': 128, 'hidden_channels': 128, 'out_channels': 64, 'dropout': 0.3, 'epochs': 250},
        {'lr': 0.005, 'embedding_dim': 64,  'hidden_channels': 256, 'out_channels': 64, 'dropout': 0.5, 'epochs': 200},
        {'lr': 0.01,  'embedding_dim': 64,  'hidden_channels': 128, 'out_channels': 32, 'dropout': 0.4, 'epochs': 150},
        {'lr': 0.001, 'embedding_dim': 256, 'hidden_channels': 256, 'out_channels': 128,'dropout': 0.3, 'epochs': 250},
        {'lr': 0.008, 'embedding_dim': 128, 'hidden_channels': 128, 'out_channels': 64, 'dropout': 0.5, 'epochs': 200},
        {'lr': 0.003, 'embedding_dim': 256, 'hidden_channels': 256, 'out_channels': 128,'dropout': 0.4, 'epochs': 150},
        {'lr': 0.005, 'embedding_dim': 64,  'hidden_channels': 64,  'out_channels': 32, 'dropout': 0.3, 'epochs': 250},
        {'lr': 0.01,  'embedding_dim': 128, 'hidden_channels': 128, 'out_channels': 64, 'dropout': 0.5, 'epochs': 200},
    ]
    results_filepath = "results/gnn_results.json"
    all_run_results = []
    
    best_run_logloss = float('inf')
    best_run_embeddings = None
    best_run_player_map = None
    
    print("\nStarting a new evaluation session...")

    for i in tqdm(range(N_RUNS), desc="Overall GNN Runs"):
        params = param_grid[i]
        run_start_time = time.time()
        
        X_placeholder = enriched_df[['tourney_date']].copy().set_index('tourney_date')
        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds, oof_true = [], []
        
        last_fold_model_state = None
        last_fold_train_graph = None
        last_fold_player_map = None
        last_fold_num_nodes = 0

        for fold, (train_index, test_index) in tqdm(enumerate(tscv.split(X_placeholder)), total=tscv.n_splits, desc=f"Run {i+1} Folds", leave=False):
            train_df, test_df = enriched_df.iloc[train_index], enriched_df.iloc[test_index]
            
            train_player_ids = pd.unique(np.concatenate([train_df['winner_id'], train_df['loser_id']]))
            player_to_idx = {pid: idx for idx, pid in enumerate(train_player_ids)}
            num_nodes = len(player_to_idx)
            
            train_df_mapped = train_df[train_df['winner_id'].isin(player_to_idx) & train_df['loser_id'].isin(player_to_idx)].copy()
            train_df_mapped['winner_idx'] = train_df_mapped['winner_id'].map(player_to_idx)
            train_df_mapped['loser_idx'] = train_df_mapped['loser_id'].map(player_to_idx)
            train_edge_np = np.vstack([train_df_mapped['winner_idx'].values, train_df_mapped['loser_idx'].values])
            train_edge_index = torch.from_numpy(train_edge_np).to(torch.long)
            train_graph = Data(num_nodes=num_nodes, edge_index=train_edge_index)

            test_df_known = test_df[test_df['winner_id'].isin(player_to_idx) & test_df['loser_id'].isin(player_to_idx)].copy()
            if not test_df_known.empty:
                test_df_known['winner_idx'] = test_df_known['winner_id'].map(player_to_idx)
                test_df_known['loser_idx'] = test_df_known['loser_id'].map(player_to_idx)
                pos_test_np = np.vstack([test_df_known['winner_idx'].values, test_df_known['loser_idx'].values])
                pos_test_idx = torch.from_numpy(pos_test_np).to(torch.long)
                neg_test_idx = negative_sampling(train_edge_index, num_nodes, pos_test_idx.size(1))
                fold_test_idx = torch.cat([pos_test_idx, neg_test_idx], dim=-1)
                fold_test_labels = torch.cat([torch.ones(pos_test_idx.size(1)), torch.zeros(neg_test_idx.size(1))], dim=0)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = GNNLinkPredictor(num_nodes, params['embedding_dim'], params['hidden_channels'], params['out_channels'], params['dropout']).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
                criterion = torch.nn.BCEWithLogitsLoss()
                
                train_graph, fold_test_idx, fold_test_labels = train_graph.to(device), fold_test_idx.to(device), fold_test_labels.to(device)

                for _ in range(params['epochs']):
                    train_one_epoch(model, train_graph, optimizer, criterion, device)
                
                _, _, fold_pred_probs = test(model, train_graph, fold_test_idx, fold_test_labels)
                oof_preds.extend(fold_pred_probs)
                oof_true.extend(fold_test_labels.cpu().numpy())

                if fold == tscv.n_splits - 1:
                    last_fold_model_state = model.state_dict()
                    last_fold_train_graph = train_graph.cpu()
                    last_fold_player_map = player_to_idx
                    last_fold_num_nodes = num_nodes

        if oof_true:
            run_logloss = log_loss(oof_true, oof_preds)
            run_duration = time.time() - run_start_time
            run_summary = {'run_id': i + 1, 'log_loss': run_logloss, 'duration_seconds': run_duration, 'params': params}
            all_run_results.append(run_summary)
            
            with open(results_filepath, 'w') as f:
                json.dump(all_run_results, f, indent=4)
            
            tqdm.write(f"-> Run {i+1} complete in {run_duration:.2f}s. Overall Log-Loss: {run_logloss:.4f}")

            if run_logloss < best_run_logloss:
                tqdm.write(f"  ** New best run found! Log-Loss: {run_logloss:.4f} **")
                best_run_logloss = run_logloss
                
                best_model_for_saving = GNNLinkPredictor(last_fold_num_nodes, params['embedding_dim'], params['hidden_channels'], params['out_channels'], params['dropout'])
                best_model_for_saving.load_state_dict(last_fold_model_state)
                best_model_for_saving.eval()
                with torch.no_grad():
                    best_run_embeddings = best_model_for_saving.encode(last_fold_train_graph.edge_index).cpu()
                best_run_player_map = last_fold_player_map

    if all_run_results:
        results_df = pd.DataFrame(all_run_results)
        gnn_scores_for_ttest = results_df['log_loss']
        print("\n--- GNN Model Final Performance Summary ---")
        print(f"Log-Loss over {len(gnn_scores_for_ttest)} runs:\n{gnn_scores_for_ttest.describe().round(4)}")

    if best_run_embeddings is not None:
        print("\nSaving artifacts from the best run...")
        torch.save(best_run_embeddings, 'results/best_gnn_embeddings.pt')
        player_map_to_save = {str(k): v for k, v in best_run_player_map.items()}
        
        with open('results/best_player_map.json', 'w') as f:
            json.dump(player_map_to_save, f, indent=4)
        print("Artifacts saved successfully.")

if __name__ == "__main__":
    run_gnn_evaluation_pipeline()