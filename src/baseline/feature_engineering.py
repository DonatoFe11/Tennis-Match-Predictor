import pandas as pd
import math
from tqdm import tqdm

def engineer_historical_features(df):
    """
    Takes a clean, sorted DataFrame and adds all historical features:
    Elo, H2H, Fatigue, and Win Streaks by iterating through the match history.
    """
    print("\n--- Section 2: Engineering Historical Features ---")
    
    # Dictionaries to hold the state for each player/pair
    elo_ratings = {}
    h2h_records = {}
    player_match_history = {}
    win_streaks = {}

    # Lists to store the calculated features for each match row
    w_elo, l_elo, w_h2h, l_h2h, w_fatigue, l_fatigue, w_streak, l_streak = [], [], [], [], [], [], [], []

    # Constants
    K_FACTOR, INITIAL_ELO, FATIGUE_WINDOW = 32, 1500, pd.Timedelta(days=7)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating historical features"):
        # Extracting match details
        w_id, l_id, date = row['winner_id'], row['loser_id'], row['tourney_date']
        duration = row['minutes'] if pd.notna(row['minutes']) else 120 # Use a default for NaN durations

        # 1. Elo Calculation
        w_elo_before = elo_ratings.get(w_id, INITIAL_ELO)
        l_elo_before = elo_ratings.get(l_id, INITIAL_ELO)
        w_elo.append(w_elo_before)
        l_elo.append(l_elo_before)
        
        # Calculate expected score and update Elo ratings
        exp_w = 1 / (1 + 10**((l_elo_before - w_elo_before) / 400))
        elo_ratings[w_id] = w_elo_before + K_FACTOR * (1 - exp_w) # increase is bigger if the win was unexpected
        elo_ratings[l_id] = l_elo_before - K_FACTOR * exp_w # decrease is bigger if the loss was unexpected
        
        # 2. H2H Calculation
        h2h_key = tuple(sorted((w_id, l_id)))
        p1_key_id, _ = h2h_key
        record = h2h_records.get(h2h_key, (0, 0)) # (p1_wins_in_key, p2_wins_in_key)
        
        w_h2h.append(record[0] if w_id == p1_key_id else record[1])
        l_h2h.append(record[1] if w_id == p1_key_id else record[0])
        
        if w_id == p1_key_id:
            h2h_records[h2h_key] = (record[0] + 1, record[1])
        else:
            h2h_records[h2h_key] = (record[0], record[1] + 1)

        # 3. Fatigue Calculation
        w_hist = player_match_history.get(w_id, [])
        l_hist = player_match_history.get(l_id, [])
        w_fatigue.append(sum(d for dt, d in w_hist if date - dt <= FATIGUE_WINDOW))
        l_fatigue.append(sum(d for dt, d in l_hist if date - dt <= FATIGUE_WINDOW))
        
        if w_id not in player_match_history: player_match_history[w_id] = []
        if l_id not in player_match_history: player_match_history[l_id] = []
        player_match_history[w_id].append((date, duration))
        player_match_history[l_id].append((date, duration))

        # 4. Win Streaks Calculation
        w_streak_before = win_streaks.get(w_id, 0)
        l_streak_before = win_streaks.get(l_id, 0)
        w_streak.append(w_streak_before)
        l_streak.append(l_streak_before)
        win_streaks[w_id] = w_streak_before + 1
        win_streaks[l_id] = 0

    # Add all new features to the dataframe so that we also have a snaphot of the storic context and of players form in that moment
    feature_lists = [w_elo, l_elo, w_h2h, l_h2h, w_fatigue, l_fatigue, w_streak, l_streak]
    col_names = ['winner_elo', 'loser_elo', 'winner_h2h', 'loser_h2h', 'winner_fatigue', 'loser_fatigue', 'winner_streak', 'loser_streak']
    for name, lst in zip(col_names, feature_lists):
        df[name] = lst
    
    print("Historical feature engineering complete.")
    return df