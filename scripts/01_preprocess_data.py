"""
Script to preprocess participant data and apply exclusion criteria.
"""
import os
import sys
import numpy as np
import pandas as pd
from src.analysis.preprocessing import (
    load_participant_data, 
    add_computed_columns, 
    exclude_participants,
    add_regressors
)
from src.utils.basic_funcs import set_seed

def main():
    # Set constants
    SEED = 2024
    set_seed(SEED)
    
    # Define paths
    DATA_FOLDER = 'data/participants/raw'
    OUTPUT_PATH = 'data/participants/trial_df.csv'
    
    # Define batches
    study1_batches = ['study1_same', 'study1_near', 'study1_far']
    study2_batches = ['study2_same', 'study2_near', 'study2_far']
    all_batches = study1_batches + study2_batches
    
    # Load and combine data
    print("Loading data...")
    df = load_participant_data(DATA_FOLDER, all_batches)
    print(f'{df["participant"].nunique()} participants total')
    
    # Add computed columns
    print("Adding computed columns...")
    df = add_computed_columns(df)
    
    # Add regressors and test trial information
    print("Adding regressors and test trial information...")
    df = add_regressors(df)
    
    # Apply exclusions
    print("Applying exclusion criteria...")
    df, trial_df = exclude_participants(df)
    
    # Print summary statistics
    print(f'{trial_df["participant"].nunique()} participants post exclusion')
    print(f'{df["participant"].nunique()-trial_df["participant"].nunique()} dropped')
    
    grouped_counts = trial_df.groupby(
        ['participant', 'condition']
    ).size().reset_index(name='count')
    print("\nParticipants per condition:")
    print(grouped_counts.groupby('condition')['participant'].nunique())
    
    # Save processed data
    print(f"\nSaving processed data to {OUTPUT_PATH}")
    trial_df.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    main()