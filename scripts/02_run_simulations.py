# Runs the simulations for the given condition settings specified in the JSON file ann_experiments.json.
# For standard simulations:
#   python scripts/02_run_simulations.py rich_50
# For geometry visualization of single participant
#   python scripts/02_run_simulations.py rich_50 --geometry
#   python scripts/02_run_simulations.py rich_50 --geometry --participant study1_same_sub20

import os
import json
import argparse
import pandas as pd
import numpy as np
from src.utils.basic_funcs import set_seed
from src.models import neural_network as net
from src.analysis import ann as ann


def load_settings(config_path):
    """Load experiment settings from JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)



def run_experiment(condition_name, base_folder='./'):
    """
    Run a single experiment condition.
    """
    # Set random seed
    set_seed(2024)
    
    # Setup paths
    data_folder = os.path.join(base_folder, 'data')
    config_path = os.path.join(base_folder, 'src', 'models', 'ann_experiments.json')
    
    # Load settings and find specified condition
    settings = load_settings(config_path)
    condition = next((c for c in settings['conditions'] if c['name'] == condition_name), None)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found in settings")
    
    # Load participant data
    df = ann.load_participant_data(data_folder)
    participants = df['participant'].unique()
    
    # Setup parameters
    task_parameters = ann.setup_task_parameters()
    
    # Network parameters
    dim_input = task_parameters['nStim_perTask'] * 2
    dim_hidden = condition['dim_hidden']
    dim_output = 4  # 2 dimensions for each feature
    network_params = [dim_input, dim_hidden, dim_output]
    
    # Training parameters - convert to list format expected by neural_network.py
    training_params = [
        participants,  # participants list
        settings['n_phase'],  # n_phase
        settings['n_epochs'],  # n_epochs
        settings['n_epochs'] * (task_parameters['nStim_perTask']*2) * 10,  # n_train_trials
        settings['shuffle'],  # shuffle
        settings['batch_size'],  # batch_size
        condition['gamma'],  # gamma
        settings['learning_rate'],  # learning rate
    ]
    
    # Setup simulation folder
    sim_folder = os.path.join(data_folder, 'simulations', condition_name)
    os.makedirs(sim_folder, exist_ok=True)
    
    # Save settings
    settings_to_save = {
        "condition": condition,
        "training_params": {
            "participants": ann.numpy_to_python(participants),
            "n_phase": settings['n_phase'],
            "n_epochs": settings['n_epochs'],
            "n_train_trials": settings['n_epochs'] * (task_parameters['nStim_perTask']*2) * 10,
            "shuffle": settings['shuffle'],
            "batch_size": settings['batch_size'],
            "gamma": condition['gamma'],
            "lr": settings['learning_rate'],
        },
        "network_params": network_params,
        "task_parameters": task_parameters
    }
    
    # Convert numpy arrays to Python native types before saving
    settings_to_save = ann.numpy_to_python(settings_to_save)
    
    with open(os.path.join(sim_folder, 'settings.json'), 'w') as f:
        json.dump(settings_to_save, f, indent=4)
    
    print(f"Starting simulation for condition: {condition_name}")
    print(f"Number of participants: {len(participants)}")
    print(f"Network architecture: input={dim_input}, hidden={dim_hidden}, output={dim_output}")
    
    # Run simulation
    net.run_simulation(
        training_params,  
        network_params,
        task_parameters,
        df,
        do_test=1,
        dosave=1,
        sim_folder=sim_folder
    )



def run_geometry_experiment(condition_name, participant_to_copy='study1_same_sub20', base_folder='./'):
    """
    Run geometry visualization experiment with matched A training.
    
    Parameters
    ----------
    condition_name : str
        Name of condition to run (e.g., 'rich_50')
    participant_to_copy : str
        ID of participant whose schedule to use for A training
    base_folder : str
        Base project folder path
    """
    # Setup paths and load settings
    data_folder = os.path.join(base_folder, 'data')
    config_path = os.path.join(base_folder, 'src', 'models', 'ann_experiments.json')
    
    settings = load_settings(config_path)
    condition = next((c for c in settings['conditions'] if c['name'] == condition_name), None)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found in settings")
    
    # Load participant data
    df = ann.load_participant_data(data_folder)
    
    # Setup parameters
    task_parameters = ann.setup_task_parameters()
    network_params = [
        task_parameters['nStim_perTask'] * 2,  # dim_input
        condition['dim_hidden'],               # dim_hidden
        4                                      # dim_output
    ]
    
    # Training parameters
    training_params = [
        [participant_to_copy],  # single participant
        settings['n_phase'],
        settings['n_epochs'],
        settings['n_epochs'] * (task_parameters['nStim_perTask']*2) * 10,
        settings['shuffle'],
        settings['batch_size'],
        condition['gamma'],
        settings['learning_rate']
    ]
    
    # Generate geometry DataFrame
    geometry_df = ann.generate_geometry_df(
        df, 
        participant_to_copy, 
        near_rule=np.pi/6, 
        far_rule=np.pi
    )
    
    # Run simulation
    print(f"Starting geometry simulation for condition: {condition_name}")
    print(f"Using participant schedule: {participant_to_copy}")
    
    geom_results = net.train_single_schedule(
        training_params,
        network_params,
        task_parameters,
        geometry_df,
        do_test=0
    )
    
    # Save results
    output_path = os.path.join(data_folder, 'simulations', f'geom_results_{condition_name}.npz')
    np.savez_compressed(output_path, **geom_results)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run neural network simulations')
    parser.add_argument('condition', type=str, 
                       help='Condition to run (e.g., rich_10, rich_50, rich_200)')
    parser.add_argument('--base-folder', type=str, default='./', 
                       help='Base project folder path (default: current directory)')
    parser.add_argument('--geometry', action='store_true',
                       help='Run geometry visualization experiment')
    parser.add_argument('--participant', type=str, default='study1_same_sub20',
                       help='Participant ID for geometry experiment')
    
    args = parser.parse_args()
    
    if args.geometry:
        run_geometry_experiment(args.condition, args.participant, args.base_folder)
    else:
        run_experiment(args.condition, args.base_folder)

if __name__ == "__main__":
    main()
