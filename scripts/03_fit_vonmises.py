# Fits von Mises distributions to participant data or simulation data.
# For participant data:
#   python scripts/03_fit_vonmises.py participants
# For simulation data:
#   python scripts/03_fit_vonmises.py simulations --sim-name rich_50
# Optional: specify base folder
#   python scripts/03_fit_vonmises.py participants --base-folder /path/to/project

import os
import sys
import numpy as np
import pandas as pd
import argparse

from src.models import vonmises as vm
from src.utils import basic_funcs as basic
from src.analysis import participant, ann, stats

def fit_human_data(participant_data):
    """Fit von mises to human data and perform model comparison"""
    # Create DataFrame with only near and far conditions
    grouped_df = (participant_data.groupby(['participant', 'condition', 'study'])[['A_rule', 'B_rule']]
                 .mean()
                 .reset_index()
                 .query("condition in ['near', 'far']"))
    
    # Initialize columns for all parameters
    sections = {
        'mixture': ['A1', 'B','A2'],  # Changed 'taskA' to 'vonmises' to match parameters dict key
        'compare': ['B','A2']
    }
    parameters = {
        'mixture': ['A_weight', 'kappa'],
        'compare': ['A_LL', 'B_LL']
    }
    
    # Initialize columns for all parameters with NaN values
    for phase, phase_sections in sections.items():
        for section in phase_sections:
            for param in parameters[phase]:
                column_name = f'{param}_{section}'
                grouped_df[column_name] = np.nan
    
    # Process each participant
    for p in grouped_df['participant'].unique():
        print(f'Processing participant {p}')
        p_data = participant_data[participant_data['participant'] == p]
        A_rule = p_data['A_rule'].iloc[0]
        B_rule = p_data['B_rule'].iloc[0]
        
        # Get responses for each task section
        responses = {
            'A1': p_data.query("task_section == 'A1' and feature_idx == 1 and block > 0")['rule_applied'].values, # fit response data after first block A
            'B': p_data.query("task_section == 'B' and feature_idx == 1 and block > 10")['rule_applied'].values,  # fit response data after first block B
            'A2': p_data.query("task_section == 'A2' and feature_idx == 1")['rule_applied'].values
        }
        
        # Fit mixture models for training phases
        for section in sections['mixture']:  
            fit_results = vm.fit_mixture_model(responses[section], A_rule, B_rule)
            for param, value in fit_results.items():
                grouped_df.loc[grouped_df['participant'] == p, f'{param}_{section}'] = value
        
        # Compare models for transfer phase
        for section in sections['compare']:
            comparison_results = vm.compare_models(responses[section], A_rule, B_rule)
            for param, value in comparison_results.items():
                grouped_df.loc[grouped_df['participant'] == p, f'{param}_{section}'] = value
    
    return grouped_df

def fit_ann_data(ann_data):
    """Fit von mises to ANN data"""

    n_stim = 6  
    
    # Create initial DataFrame with conditions
    participants = []
    conditions = []
    for condition in ['near', 'far']:
        for i in range(len(ann_data[condition])):
            participants.append(str(ann_data[condition][i]['participant']))
            conditions.append(condition)
    
    grouped_df = pd.DataFrame({
        'participant': participants,
        'condition': conditions
    })
    
    # Initialize columns
    for section in ['A1', 'B', 'A2']:
        grouped_df[f'A_weight_{section}'] = np.nan
        grouped_df[f'kappa_{section}'] = np.nan
    
    for s_idx, schedule_data in enumerate([ann_data['near'], ann_data['far']]):
        for subj in range(len(schedule_data)):
            print(f'Processing ANN participant {subj}')
            
            # Calculate rules
            ruleA = np.arctan2(schedule_data[subj]['labels'][0,1,:][0], schedule_data[subj]['labels'][0,1,:][1]) - \
                    np.arctan2(schedule_data[subj]['labels'][0,0,:][0], schedule_data[subj]['labels'][0,0,:][1])
            ruleB = np.arctan2(schedule_data[subj]['labels'][1,1,:][0], schedule_data[subj]['labels'][1,1,:][1]) - \
                    np.arctan2(schedule_data[subj]['labels'][1,0,:][0], schedule_data[subj]['labels'][1,0,:][1])
            
            # Get responses for each section
            responses = {}
            for task_section_idx, section in enumerate(['A1', 'B', 'A2']):
                summer_radians = np.arctan2(schedule_data[subj]['predictions'][task_section_idx,::2,0],
                                          schedule_data[subj]['predictions'][task_section_idx,::2,1])
                winter_radians = np.arctan2(schedule_data[subj]['predictions'][task_section_idx,1::2,2],
                                          schedule_data[subj]['predictions'][task_section_idx,1::2,3])
                
                response_angle = winter_radians - summer_radians
                response_angle = basic.wrap_to_pi(response_angle) 

                # Each section, ANNs are trained on 100 iterations of participant training data (10 blocks x 6 stimuli x 100 iterations)
                # To match human data length i.e. 60 data points per section, we take first block (six stimuli) on every tenth iteration
                responses[section] = np.concatenate([response_angle[i:i+n_stim] 
                                                                         for i in range(0,len(response_angle),n_stim*100)])
            
            # Fit mixture models
            participant_id = str(schedule_data[subj]['participant'])
            for section in ['A1', 'B', 'A2']:
                fit_results = vm.fit_mixture_model(responses[section], ruleA, ruleB)
                grouped_df.loc[grouped_df['participant']==participant_id, f'A_weight_{section}'] = fit_results['A_weight']
                grouped_df.loc[grouped_df['participant']==participant_id, f'kappa_{section}'] = fit_results['kappa']
    
    return grouped_df

def run_analysis(data_type, sim_name=None, base_folder='./'):
    """
    Run von Mises analysis for specified data type.
    
    Parameters
    ----------
    data_type : str
        Type of data to analyze ('participants' or 'simulations')
    sim_name : str, optional
        Name of simulation folder (required if data_type is 'simulations')
    base_folder : str
        Base project folder path
    """
    # Setup paths
    data_folder = os.path.join(base_folder, 'data')
    
    if data_type == 'participants':
        # Load and fit participant data
        participant_data = pd.read_csv(os.path.join(data_folder, 'participants', 'trial_df.csv'))
        grouped_df = fit_human_data(participant_data)
        output_path = os.path.join(data_folder, 'participants', 'human_vonmises_fits.csv')
        
    elif data_type == 'simulations':
        if not sim_name:
            raise ValueError("Simulation name must be provided for simulation data")
        # Load and fit simulation data
        ann_data = ann.load_ann_data(os.path.join(data_folder, 'simulations', sim_name))
        grouped_df = fit_ann_data(ann_data)
        output_path = os.path.join(data_folder, 'simulations', f'{sim_name}_vonmises_fits.csv')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    grouped_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Fit von Mises distributions to participant or simulation data')
    parser.add_argument('data_type', choices=['participants', 'simulations'],
                      help='Type of data to analyze')
    parser.add_argument('--sim-name', type=str,
                      help='Name of simulation folder (required if data_type is simulations)')
    parser.add_argument('--base-folder', type=str, default='./',
                      help='Base project folder path (default: current directory)')
    
    args = parser.parse_args()
    run_analysis(args.data_type, args.sim_name, args.base_folder)

if __name__ == "__main__":
    main()