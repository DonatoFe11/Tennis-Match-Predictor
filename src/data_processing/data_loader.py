import pandas as pd
import glob
import re
import numpy as np

def load_and_prepare_data(path_pattern='data/tennis_atp/atp_matches_*.csv', start_year=2000):
    """
    Loads all ATP match data from a given year onwards, cleans basic data types,
    and sorts the data chronologically.
    """
    print("--- Section 1: Loading and Preparing Raw Data ---")
    
    all_files = glob.glob(path_pattern)
    year_pattern = re.compile(r'atp_matches_(\d{4})\.csv$')
    
    all_files_filtered = []
    for f in all_files:
        match = year_pattern.search(f)
        if match and int(match.group(1)) >= start_year:
            all_files_filtered.append(f)

    if not all_files_filtered:
        raise FileNotFoundError(f"No match files found for pattern {path_pattern} from year {start_year} onwards.")

    print(f"Loading {len(all_files_filtered)} files from year {start_year} onwards...")
    li = [pd.read_csv(filename, index_col=None, header=0, low_memory=False) for filename in sorted(all_files_filtered)]
    data = pd.concat(li, axis=0, ignore_index=True)
    
    print(f"Loaded {len(data)} matches.")

    # Convert tourney_date to datetime objects
    data['tourney_date'] = pd.to_datetime(data['tourney_date'], format='%Y%m%d')

    # Ensure all relevant statistical columns are numeric, coercing errors to NaN
    stat_cols = [
        'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 
        'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
    ]
    for col in stat_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            # If a stat column is missing entirely, add it with NaNs for consistency
            data[col] = np.nan
    
    # Sort and return
    print("Data loading and basic preparation complete.")
    return data.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)