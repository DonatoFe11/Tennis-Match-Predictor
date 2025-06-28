import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

def create_graph_data(df, test_year=2023):
    """
    Creates the full graph data object and performs a temporal split.
    """
    print("\n--- Section 3: Building Graph and Splitting Data ---")

    # --- 1. Node Mapping from TRAINING data ONLY ---
    train_df = df[df['tourney_date'].dt.year < test_year]
    all_player_ids = pd.unique(np.concatenate([train_df['winner_id'], train_df['loser_id']]))
    player_to_idx = {player_id: i for i, player_id in enumerate(all_player_ids)}
    num_train_nodes = len(player_to_idx)
    print(f"Number of unique players in the training period (nodes): {num_train_nodes}")

    # --- 2. Build the TRAINING Graph ---
    train_df = train_df.copy()
    train_df['winner_idx'] = train_df['winner_id'].map(player_to_idx)
    train_df['loser_idx'] = train_df['loser_id'].map(player_to_idx)
    train_df.dropna(subset=['winner_idx', 'loser_idx'], inplace=True)
    
    train_edge_index = torch.tensor([
        train_df['winner_idx'].values,
        train_df['loser_idx'].values
    ], dtype=torch.long)
    
    train_graph_data = Data(num_nodes=num_train_nodes, edge_index=train_edge_index)
    print("Training Graph created:", train_graph_data)

    # --- 3. Prepare the TEST Set (Positive and Negative Edges) ---
    test_df = df[df['tourney_date'].dt.year >= test_year].copy()
    test_df_known = test_df[
        test_df['winner_id'].isin(player_to_idx) & 
        test_df['loser_id'].isin(player_to_idx)
    ]
    test_df_known['winner_idx'] = test_df_known['winner_id'].map(player_to_idx)
    test_df_known['loser_idx'] = test_df_known['loser_id'].map(player_to_idx)

    pos_test_edge_index = torch.tensor([
        test_df_known['winner_idx'].values,
        test_df_known['loser_idx'].values
    ], dtype=torch.long)
    pos_test_edge_label = torch.ones(pos_test_edge_index.size(1))

    neg_test_edge_index = negative_sampling(
        edge_index=train_graph_data.edge_index,
        num_nodes=train_graph_data.num_nodes,
        num_neg_samples=pos_test_edge_index.size(1)
    )
    neg_test_edge_label = torch.zeros(neg_test_edge_index.size(1))

    test_edge_label_index = torch.cat([pos_test_edge_index, neg_test_edge_index], dim=-1)
    test_edge_label = torch.cat([pos_test_edge_label, neg_test_edge_label], dim=0)
    
    print(f"Test edges prepared: {test_edge_label_index.shape[1]} total edges for evaluation.")
    
    return train_graph_data, test_edge_label_index, test_edge_label, player_to_idx, num_train_nodes

