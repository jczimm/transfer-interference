
import pandas as pd
import numpy as np
from scipy import stats

def compute_transfer_humans(trial_df):
    """
    Calculate transfer metrics for human data.
    """
  
     # Get final 6 trials of A1 
    A1_accuracy = trial_df.loc[(trial_df['task_section']=='A1') & (trial_df['feature_idx']==1), :]
    final_a1 = A1_accuracy.groupby(['participant', 'condition', 'study']).agg(
            accuracy_A1=('accuracy', lambda x: np.mean(x.values[-6:]))
        ).reset_index()
    # Get first 6 trials of B
    B_accuracy = trial_df.loc[(trial_df['task_section']=='B') & (trial_df['feature_idx']==1), :]
    first_b = B_accuracy.groupby(['participant', 'condition', 'study']).agg(
            accuracy_B=('accuracy', lambda x: np.mean(x.values[0:6])) 
        ).reset_index()

    # Calculate transfer metrics
    transfer_df = pd.merge(
            final_a1, 
            first_b, 
            on=['participant', 'condition', 'study'], 

        )
    transfer_df['error_diff'] = transfer_df['accuracy_B'] - transfer_df['accuracy_A1']

    return transfer_df


def add_behav_metrics(group_df, trial_df):

    # define lumpers and splitters
    group_df['is_lumper'] = (group_df['B_LL_A2'] > group_df['A_LL_A2'])
    group_df['group'] = np.where(group_df['is_lumper']==1, 'lumpers', 'splitters')

    new_cols = ['summer_accuracy', 'transfer_error_diff', 'generalisation_acc', 'interference', 'correct_afc','retest_error_diff']
    group_df[new_cols] = np.nan

    # Calculate metrics for each participant
    for p in group_df['participant'].unique():

        p_data = trial_df[trial_df['participant']==p]

        group_df.loc[group_df['participant']==p, 'interference']  =  1-group_df.loc[group_df['participant']==p, 'A_weight_A2'].values[0].astype(np.float32) # interference = use of B rule at A2 
             

        # Add summer accuracy
        group_df.loc[group_df['participant']==p, 'summer_accuracy'] = p_data[p_data['feature_idx']==0]['accuracy'].mean()
        
        # Add transfer 
        final_A1 = p_data[(p_data['task_section']=='A1') & (p_data['feature_idx']==1)].iloc[-6:]['accuracy'].mean()
        initial_B = p_data[(p_data['task_section']=='B') & (p_data['feature_idx']==1)].iloc[:6]['accuracy'].mean()
        A2_accuracy = p_data[(p_data['task_section']=='A2') & (p_data['feature_idx']==1)]['accuracy'].mean()
        group_df.loc[group_df['participant']==p, 'transfer_error_diff'] = initial_B - final_A1
        group_df.loc[group_df['participant']==p, 'retest_error_diff'] =  A2_accuracy - final_A1

        # Add generalisation accuracy
        group_df.loc[group_df['participant']==p, 'generalisation_acc'] = p_data[(p_data['test_trial']==1) & 
                                                                            (p_data['task_section']=='A1') & 
                                                                            (p_data['block']>=5)]['accuracy'].mean() # find generalisation accuracy second half of A1

    # Add debrief data
    afc_dat = trial_df[trial_df['task_section']=='debrief'].groupby('participant')['correct_afc'].mean().reset_index()
    afc_dat['correct_afc'] = 100 * afc_dat['correct_afc']

    if 'correct_afc' in group_df.columns:
        group_df = group_df.drop(columns=['correct_afc'])
    group_df = pd.merge(group_df, afc_dat, on=['participant'])

    return group_df 