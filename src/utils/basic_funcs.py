"""
Basic utility functions 
"""
import numpy as np
import random
import torch

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def get_datasets(df, participant, task_parameters):
    """
    Main function to get datasets and process raw inputs and labels.
    """
    # Filter data for each task section
    participant_training_A1 = filter_participant_data(df, participant, 'A1')
    participant_training_B = filter_participant_data(df, participant, 'B')
    participant_training_A2 = filter_participant_data(df, participant, 'A2')

    # Adjust indices for B and C
    A_length = len(participant_training_A1)
    B_length = len(participant_training_B)
    participant_training_B = adjust_indices(participant_training_B, A_length)
    participant_training_A2 = adjust_indices(participant_training_A2, A_length + B_length)

    # Create inputs matrices
    A1_inputs = create_inputs_matrix(participant_training_A1, task_parameters['nStim_perTask'])
    B_inputs = create_inputs_matrix(participant_training_B, task_parameters['nStim_perTask'])
    A2_inputs = create_inputs_matrix(participant_training_A2, task_parameters['nStim_perTask'])

    # Process raw inputs and labels
    raw_inputs = np.full((3, task_parameters['nStim_perTask'], task_parameters['nStim_perTask'] * 2), np.nan, dtype=np.float32)
    raw_labels = np.full((3, 4, task_parameters['nStim_perTask']), np.nan, dtype=np.float32)

    raw_inputs[0], raw_labels[0] = process_raw_inputs_and_labels(participant_training_A1, task_parameters['nStim_perTask'], 0)
    raw_inputs[1], raw_labels[1] = process_raw_inputs_and_labels(participant_training_B, task_parameters['nStim_perTask'], 1)
    raw_inputs[2], raw_labels[2] = process_raw_inputs_and_labels(participant_training_A2, task_parameters['nStim_perTask'], 2)

    # Assemble datasets
    dataset_A1 = assemble_dataset(participant_training_A1, A1_inputs, np.cos(participant_training_A1['feat_val'].values), np.sin(participant_training_A1['feat_val'].values))
    dataset_B = assemble_dataset(participant_training_B, B_inputs, np.cos(participant_training_B['feat_val'].values), np.sin(participant_training_B['feat_val'].values))
    dataset_A2 = assemble_dataset(participant_training_A2, A2_inputs, np.cos(participant_training_A2['feat_val'].values), np.sin(participant_training_A2['feat_val'].values))

    return dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels


def filter_participant_data(df, participant, task_section):
    """
    Filter participant data by task section.
    """
    return df.loc[
        (df['participant'] == participant) & (df['task_section'] == task_section),
        ['index', 'feature_idx', 'feat_val', 'noisy_feedback_value', 'stimID','test_trial']
    ].reset_index(drop=True)


def adjust_indices(participant_data, offset):
    """
    Adjust the indices of participant data by the specified offset.
    """
    participant_data['index'] -= offset
    return participant_data.reset_index(drop=True)

def create_inputs_matrix(participant_data, n_stim_per_task):
    """
    Create an inputs matrix with one-hot encoded stimulus IDs.
    """
    length = participant_data.shape[0]
    inputs = np.zeros((length, n_stim_per_task * 2))
    for index, row in participant_data.iterrows():
        inputs[index, int(row['stimID'])] = 1
    return inputs

def process_raw_inputs_and_labels(participant_data, n_stim_per_task, task_idx):
    """
    Process raw inputs and labels for a given task.
    """
    unique_inputs = participant_data['stimID'].unique().astype(int)
    raw_inputs = np.full((n_stim_per_task, n_stim_per_task * 2), np.nan, dtype=np.float32)
    raw_labels = np.full((4, n_stim_per_task), np.nan, dtype=np.float32)

    for idx, stim_id in enumerate(unique_inputs):
        feat1 = participant_data.loc[
            (participant_data['stimID'] == stim_id) & (participant_data['feature_idx'] == 0), 'feat_val'
        ].unique()
        feat2 = participant_data.loc[
            (participant_data['stimID'] == stim_id) & (participant_data['feature_idx'] == 1), 'feat_val'
        ].unique()
        raw_labels[0, idx] = np.cos(feat1)[0]
        raw_labels[1, idx] = np.sin(feat1)[0]
        raw_labels[2, idx] = np.cos(feat2)[0]
        raw_labels[3, idx] = np.sin(feat2)[0]

        input_skeleton = np.zeros((n_stim_per_task * 2))
        input_skeleton[stim_id] = 1
        raw_inputs[idx, :] = input_skeleton

    return raw_inputs, raw_labels

def assemble_dataset(participant_data, inputs, label_cos, label_sin):
    """
    Assemble the dataset dictionary for a task.
    """
    return {
        'index': participant_data['index'].values,
        'stim_index': participant_data['stimID'].values,
        'input': inputs,
        'feature_probe': participant_data['feature_idx'].values,
        'test_stim': participant_data['test_trial'].values,
        'label_x': label_cos,
        'label_y': label_sin,
    }

def get_clockwise_order(labels):
    """
    Provides an ordering of labels starting at the top of a circle and moving clockwise.

    Args:
    labels: np.ndarray
        A 6x2 matrix representing X-Y coordinates for each stimulus.

    Returns:
    list
        Indices of labels in the clockwise order starting from the top.
    """
    # Extract X and Y coordinates
    X = labels[:, 0]
    Y = labels[:, 1]

    # Convert to radians using arctan2 (Y, X)
    angles = np.arctan2(Y, X)

    # Convert angles from range (-pi, pi) to (0, 2*pi) with 0 being at the positive Y-axis
    angles = np.mod(-angles + np.pi / 2, 2 * np.pi)

    # Get the order of indices by sorting the angles in ascending order
    ordered_indices = np.argsort(angles)

    return ordered_indices


def wrap_to_pi(values):
    """
    Wrap angles to the range [-pi, pi].
    
    Args:
        values (float, np.ndarray, pd.Series): Angle(s) in radians.

    Returns:
        Wrapped angle(s) in the range [-pi, pi].
    """
    return (values + np.pi) % (2 * np.pi) - np.pi