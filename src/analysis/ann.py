import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from src.utils import basic_funcs as basic


def load_ann_data(ann_folder):
    """
    Load ANN simulation data from npz files and organize by condition.
    
    Args:
        ann_folder (str): Path to folder containing simulation results
            (e.g., 'data/simulations/rich_50')
            
    Returns:
        dict: Dictionary with keys 'same', 'near', 'far' containing simulation data
              for each condition
    """
    # Initialize containers for each condition
    condition_data = {
        'same': [],
        'near': [],
        'far': []
    }

    # Loop through files in the simulation folder
    for file_name in os.listdir(ann_folder):
        if not file_name.endswith('.npz'):
            continue
            
        file_path = os.path.join(ann_folder, file_name)
        
        # Sort into appropriate condition list
        for condition in condition_data.keys():
            if condition in file_name:
                with np.load(file_path, allow_pickle=True) as data:  # Ensure file is closed after reading

                    condition_data[condition].append({
                        'participant': file_name.replace('.npz', ''),
                        'predictions': data['predictions'],   
                        'labels': data['labels'],
                        'accuracy': data['accuracy'],
                        'losses': data['losses'],
                        'test_stim': data['test_stim'],
                        'hiddens_post_phase_0': data['hiddens_post_phase_0'],
                        'hiddens_post_phase_1': data['hiddens_post_phase_1'],
                    })
                break  
                
    return condition_data


# functions for preparing participant data for ANN training
def setup_task_parameters():
    """Define basic task parameters."""
    return {
        "nStim_perTask": 6,
        "schedules": ['same', 'near', 'far'],
        "schedule_names": ['same rule', 'near rule', 'far rule']
    }

def load_participant_data(data_folder):
    """Load participant data for ANN training."""
    df = pd.read_csv(os.path.join(data_folder, 'participants', 'trial_df.csv'))
    df.loc[df['task_section']=='B','test_trial']=0
    return df.loc[(df['task_section']=='A1') | 
                 (df['task_section']=='B') | 
                 (df['task_section']=='A2'), :] # remove debrief trials from analysis

def numpy_to_python(obj):
    """Convert numpy objects to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(item) for item in obj]
    return obj

def generate_geometry_df(df, participant_to_copy, near_rule=np.pi/6, far_rule=np.pi):
    """
    Create phantom DataFrame where A training is matched, for the purpose of visualising representational geometry changes
    
    Parameters
    ----------
    df : pd.DataFrame
        Original participant DataFrame
    participant_to_copy : str
        ID of participant whose schedule to use for A training
    near_rule : float, optional
        Rule adjustment for near condition (default: pi/6)
    far_rule : float, optional
        Rule adjustment for far condition (default: pi)
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with matched A training and adjusted rules for each condition
    """
    conditions = {'same': 0, 'near': near_rule, 'far': far_rule}
    combined_dfs = []

    for condition, rule_adjustment in conditions.items():
        temp_df = df[df['participant'] == participant_to_copy].copy()
        temp_df['participant'] = f"geom_sub_{condition}"
        temp_df['condition'] = condition
        if condition != 'same':
            temp_df = adjust_rule_and_feat_val(temp_df, rule_adjustment)
        combined_dfs.append(temp_df)
    
    return pd.concat(combined_dfs, ignore_index=True)

def adjust_rule_and_feat_val(df_subset, rule_adjustment):
    """
    Adjust B_rule and feat_val for circular wrapping.
    
    Args:
        df_subset (pd.DataFrame): Dataframe to adjust.
        rule_adjustment (float): Angle adjustment value in radians.

    Returns:
        pd.DataFrame: Adjusted dataframe.
    """
    df_copy = df_subset.copy()
    # Adjust B_rule
    df_copy['B_rule'] = basic.wrap_to_pi(df_copy['B_rule'] + rule_adjustment)
    # Adjust feat_val for task_section == 'B' and feature_idx == 1
    mask = (df_copy['task_section'] == 'B') & (df_copy['feature_idx'] == 1)
    df_copy.loc[mask, 'feat_val'] = basic.wrap_to_pi(df_copy.loc[mask, 'feat_val'] + rule_adjustment)
    return df_copy


# functions for analysing ANN simulation results
def compute_transfer_anns(ann_data):
    """
    Calculate transfer/switch cost metrics for ANN data.
    
    Args:
        ann_data (dict): Dictionary containing ANN simulation results
            
    Returns:
        pd.DataFrame: Transfer metrics with columns:
            - participant
            - condition
            - transfer_error_diff
    """
    agg_data = []

    for schedule_name, schedule_data in ann_data.items():
        for subj in range(len(schedule_data)):
            # Get accuracy by task section
            A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy() # winter responses only
            B_accuracy = schedule_data[subj]['accuracy'][1, 1::2].copy() # winter responses only
            
            # Average over relevant window
            final_A1_acc = np.mean(A1_accuracy[-6:])  # final pass through A stim (winter)
            initial_B_acc = np.mean(B_accuracy[0:6])  # first pass through B stim (winter) 
            
            # Calculate transfer cost
            error_diff = initial_B_acc - final_A1_acc
            
            # Store results
            agg_data.append({
                'participant': str(schedule_data[subj]['participant']), 
                'condition': schedule_name, 
                'error_diff': error_diff
            })
    
    return pd.DataFrame(agg_data)





def analyze_training_loss(ann_data, save_path=None):
    """
    Analyze and optionally save training loss curves for all schedules.
    
    Parameters
    ----------
    ann_data : dict
        Dictionary containing ANN data
    save_path : str, optional
        Path to save figures
        
    Returns
    -------
    dict
        Dictionary containing loss statistics for each schedule
    """
    results = {}
    
    for schedule_name in ['same', 'near', 'far']:
        schedule_data = ann_data[schedule_name]
        
        # Calculate loss statistics
        flattened_length = schedule_data[0]['losses'].shape[0] * schedule_data[0]['losses'].shape[1]
        sched_losses = np.zeros((len(schedule_data), flattened_length))
        
        for subj in range(len(schedule_data)):
            flat_loss = np.concatenate(schedule_data[subj]['losses'], axis=0)
            sched_losses[subj, :] = flat_loss
        
        results[schedule_name] = {
            'mean': np.nanmean(sched_losses, axis=0)[1::2],
            'std': np.nanstd(sched_losses, axis=0)[1::2]
        }
    
    return results


def compute_pca_components(ann_data, variance_threshold=0.99):
    """
    Compute the number of PCA components needed to explain a threshold of the variance
    in hidden layer representations after tasks A and B.
    
    Parameters:
    -----------
    ann_data : dict
        Dictionary containing simulation data for different schedules.
    variance_threshold : float, optional
        Threshold for cumulative explained variance ratio (default: 0.99).
        
    Returns:
    --------
    pandas.DataFrame
        Long-format DataFrame containing PCA components for each task and condition.
    """
    results = {
        'participant': [],
        'condition': [],
        'task': [],
        'n_pca': []
    }
    
    for schedule_name, schedule_data in ann_data.items():
        for subj in range(len(schedule_data)):
            # Get hidden activity for A and B
            A_hids = schedule_data[subj]['hiddens_post_phase_0']
            B_hids = schedule_data[subj]['hiddens_post_phase_1']
            
            # Fit PCA
            pca_A_full = PCA().fit(A_hids)
            pca_B_full = PCA().fit(B_hids)
            
            # Find number of components needed for variance threshold
            n_components_A = np.argmax(np.cumsum(pca_A_full.explained_variance_ratio_) >= variance_threshold) + 1
            n_components_B = np.argmax(np.cumsum(pca_B_full.explained_variance_ratio_) >= variance_threshold) + 1
            
            # Store results
            participant_id = str(schedule_data[subj]['participant'])
            results['participant'].append(participant_id)
            results['condition'].append(schedule_name)
            results['task'].append('post A')
            results['n_pca'].append(n_components_A)
            
            results['participant'].append(participant_id)
            results['condition'].append(schedule_name)
            results['task'].append('post B')
            results['n_pca'].append(n_components_B)
    
    # Convert results to DataFrame
    agg_df_long = pd.DataFrame(results)
    
    return agg_df_long

def get_principal_angles(ann_data):
    """
    Parameters:
    -----------
    ann_data : dict
        Dictionary containing simulation data for different schedules.
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing participant IDs, conditions, and computed principal angles.
    """
    # Initialize results dictionary
    results = {
        'participant': [],
        'condition': [],
        'principal_angle_between': []
    }

    # Iterate through conditions and participants
    for schedule_name, schedule_data in ann_data.items():
        
        for subj in range(len(schedule_data)):

            # Extract hidden representations – already ordered by A and B task inputs
            A_hids = schedule_data[subj]['hiddens_post_phase_1'][0:6, :].copy()
            B_hids = schedule_data[subj]['hiddens_post_phase_1'][6:, :].copy()
            
            # Compute principal angle between A and B representations
            angle_between, _ = compute_principal_angle(A_hids, B_hids, n_components=2)
            
            # Store results
            results['participant'].append(str(schedule_data[subj]['participant']))
            results['condition'].append(schedule_name)
            results['principal_angle_between'].append(angle_between)
    
    # Create DataFrame from results
    agg_df = pd.DataFrame(results)
    
    return agg_df

def compute_principal_angle(A_hids, B_hids, n_components=2):
    """
    Compute the principal angles between subspaces spanned by A_hids and B_hids.
    
    Args:
        A_hids (numpy.ndarray): Hidden representations for task A stimuli, shape (n_samples_A, in_weights).
        B_hids (numpy.ndarray): Hidden representations for task B stimuli, shape (n_samples_B, in_weights).
        n_components (int): Number of principal components to consider for the subspaces.
    
    Returns:
        float: The first principal angle (in degrees).
        numpy.ndarray: All principal angles (in degrees).
    """
    # Step 1: Fit PCA to both sets of hidden representations
    pca_A = PCA(n_components=n_components)
    pca_B = PCA(n_components=n_components)

    V_A = pca_A.fit_transform(A_hids)  # Principal components for A
    V_B = pca_B.fit_transform(B_hids)  # Principal components for B

    # Step 2: Compute the inner product matrix between the two PCA bases
    inner_product_matrix = np.dot(pca_A.components_, pca_B.components_.T)

    # Step 3: Perform SVD on the inner product matrix
    _, singular_values, _ = np.linalg.svd(inner_product_matrix)

    # Step 4: Compute the principal angles
    principal_angles = np.arccos(np.clip(singular_values, -1.0, 1.0))  # Ensure numerical stability

    # Convert to degrees for interpretability
    principal_angles_degrees = np.degrees(principal_angles)

    # Return the first principal angle and all angles
    return principal_angles_degrees[0], principal_angles_degrees



def prepare_pca_single_task(hids):
    # Fit PCA on Task B's hidden layer after learning
    pca = PCA(n_components=2)
    pca.fit(hids)
    task_transformed = pca.transform(hids)
    return pca, task_transformed

def project_onto_pca(pca, data):
    # Project data onto the PCA space defined by pca
    return pca.transform(data)


def get_hiddens(geom_results):
    # A is same for each condition in geom_results
    hiddens_postA = geom_results['hiddens_post_phase_0'][0, :, :]

    # get separate B hiddens for each condition
    same_hiddens_postB = geom_results['hiddens_post_phase_1'][0, :, :]
    near_hiddens_postB = geom_results['hiddens_post_phase_1'][1, :, :]
    far_hiddens_postB = geom_results['hiddens_post_phase_1'][2, :, :]

    return hiddens_postA, same_hiddens_postB, near_hiddens_postB, far_hiddens_postB




## Individual differences in lazy/rich analyses

def add_ann_metrics(rich_data, lazy_data, rich_group_params, lazy_group_params):
# Get aggregated data

    ann_behav_data = []

    for dat_name,schedule_data,group_params in zip(['rich','lazy'],[rich_data, lazy_data],[rich_group_params, lazy_group_params]):
        
            
        final_A_acc = np.full((len(schedule_data)),np.nan)
        initial_B_acc = np.full((len(schedule_data)),np.nan)

        # Loop through participants and save their flattened losses
        for subj in range(len(schedule_data)):

            # get accuracy by task section
            A1_accuracy = schedule_data[subj]['accuracy'][0, 1::2].copy() # winter only i.e. odd features
            B_accuracy = schedule_data[subj]['accuracy'][1, 1::2].copy()
            A2_accuracy = schedule_data[subj]['accuracy'][2, 1::2].copy() 
                    
            # average over relevant window
            final_A1_acc = np.mean(A1_accuracy[-6:]) # final pass through A stimuli set 
            initial_B_acc = np.mean(B_accuracy[0:6]) # first pass through B stimuli set 
            A2_accuracy = np.mean(A2_accuracy) 
                    
            # get diffs
            transfer_error_diff = initial_B_acc -final_A1_acc
            retest_error_diff = A2_accuracy - final_A1_acc
                    
            # summer accuracy
            summer_accuracy = np.mean(schedule_data[subj]['accuracy'][0, 0::2].copy())

            # generalisation accuracy
            test_stim = schedule_data[subj]['test_stim'][0,1::2].copy().astype(int)
            all_A1_accuracy =  schedule_data[subj]['accuracy'][0,1::2].copy() # winter only
            all_A1_accuracy[test_stim==0]=np.nan 
            generalisation_accuracy = np.nanmean(all_A1_accuracy)
                    
            retest_int = 1- group_params.loc[group_params['participant']==str(schedule_data[subj]['participant']),f'A_weight_A2'].values[0].astype(np.float32) # interference = use of B rule at A2 
                    
            # Append data 
            ann_behav_data.append({'group': dat_name,'participant': str(schedule_data[subj]['participant']), 'initialB': initial_B_acc,'transfer_error_diff': transfer_error_diff, 'retest_error_diff': retest_error_diff, 'summer_accuracy': summer_accuracy, 'generalisation_acc': generalisation_accuracy, 'interference': retest_int})

    ann_behav_df = pd.DataFrame(ann_behav_data)

    return ann_behav_df
