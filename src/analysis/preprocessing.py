"""
Functions for preprocessing participant data.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

def load_participant_data(data_folder, batches):
    """Load and combine participant data from CSV files."""
    all_data = []
    
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.csv'):
            for batch in batches:
                if file_name.startswith(batch):
                    file_path = os.path.join(data_folder, file_name)
                    tmp = pd.read_csv(file_path)
                    tmp['regime'] = batch
                    all_data.append(tmp)
    
    return pd.concat(all_data, ignore_index=True)

def add_computed_columns(df, nStim=6):
    """Add computed columns to the dataframe."""
    df['accuracy'] = 1-(df['resp_error']/np.pi)
    df['condition'] = [item.split('_')[-1] for item in df['regime']]
    df['study'] = df['participant'].apply(
        lambda x: 1 if x.startswith('study1') else (2 if x.startswith('study2') else None)
    )
    return df

def exclude_participants(df, chance_error=np.pi/2):
    """Apply exclusion criteria to participants."""
    # Create clean copy
    trial_df = df.copy()
    
    # Drop participants who admit using pen
    trial_df = trial_df[trial_df['debrief_tools'] != 'yes-pen']
    
    # Mark participants for exclusion based on performance
    df['excluded'] = 0
    participants_to_exclude = []

    for p in trial_df['participant'].unique():
        taskA_error = trial_df.loc[
            (trial_df['task_section'] == 'A1') & 
            (trial_df['participant'] == p), 'resp_error'
        ].values
        taskA_error_final = taskA_error[-24:]  # final 2 blocks of training task A 
        
        pvalue = stats.ttest_1samp(
            taskA_error_final, 
            chance_error, 
            alternative='less'
        ).pvalue
        
        if pvalue > 0.05:
            participants_to_exclude.append(p)
    
    # Update exclusion flags and create clean dataset
    df.loc[df['participant'].isin(participants_to_exclude), 'excluded'] = 1
    trial_df = trial_df[~trial_df['participant'].isin(participants_to_exclude)]
    
    return df, trial_df


def calculate_rules(group):
    """Calculate A and B rules for a group."""
    # Calculate A rule
    A_rules = group.loc[(group['feature_idx'] == 1) & (group['task_section'] == 'A1'), 'feat_val'].values \
              - group.loc[(group['feature_idx'] == 0) & (group['task_section'] == 'A1'), 'feat_val'].values
    A_rules = (A_rules + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    taskA_rules = np.squeeze(np.unique(np.round(A_rules, 6)))

    # Calculate B rule
    B_rules = group.loc[(group['feature_idx'] == 1) & (group['task_section'] == 'B'), 'feat_val'].values \
              - group.loc[(group['feature_idx'] == 0) & (group['task_section'] == 'B'), 'feat_val'].values
    B_rules = (B_rules + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    taskB_rules = np.squeeze(np.unique(np.round(B_rules, 6)))

    return taskA_rules, taskB_rules

def add_test_stim(df):
    """Add test stimulus information to dataframe."""
    # Find test stimulus ID for each participant and add it to all rows
    df['test_stim_B'] = np.nan
    df['test_stim_A'] = np.nan

    test_stim_A = df.loc[(df['noisy_feedback_value'].isna()) & (df['task_section'] == 'A1'), 
                         ['participant', 'stimID']].drop_duplicates()
    test_stim_B = df.loc[(df['noisy_feedback_value'].isna()) & (df['task_section'] == 'B'), 
                         ['participant', 'stimID']].drop_duplicates()

    # Use map to assign test stimulus values to participants
    df['test_stim_A'] = df['participant'].map(test_stim_A.set_index('participant')['stimID'])
    df['test_stim_B'] = df['participant'].map(test_stim_B.set_index('participant')['stimID'])

    # Add test trial indicator
    df['test_trial'] = 0
    df.loc[(df['test_stim_A'] == df['stimID']) & (df['feature_idx'] == 1), 'test_trial'] = 1
    df.loc[(df['test_stim_B'] == df['stimID']) & (df['feature_idx'] == 1), 'test_trial'] = 1

    return df

def add_regressors(df):
    """Add additional regressors to dataframe."""
    # Add 'A_rule' and 'B_rule' columns
    df_groupstim = df.groupby(['participant', 'stimID', 'feature_idx', 'task_section'])[['noisy_feedback_value','feat_val']].mean().reset_index()

    df['A_rule'] = np.nan
    df['B_rule'] = np.nan

    # Calculate A_rule and B_rule for each participant
    grouped = df_groupstim.groupby('participant')
    rules = grouped.apply(calculate_rules)

    # Apply the calculated rules to the original dataframe
    for p, (a_rule, b_rule) in rules.items():
        df.loc[df['participant'] == p, 'A_rule'] = a_rule
        df.loc[df['participant'] == p, 'B_rule'] = b_rule

    # Calculate 'rule_applied' using vectorized operations
    rule_applied = df.loc[df['feature_idx'] == 1, 'dial_resp'].values \
                   - df.loc[df['feature_idx'] == 0, 'noisy_feedback_value'].values
    rule_applied = (rule_applied + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
    df.loc[df['feature_idx'] == 1, 'rule_applied'] = rule_applied

    # Add test stim columns
    df = add_test_stim(df)

    return df