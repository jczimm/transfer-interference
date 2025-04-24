"""
Neural network models

This module contains the core functionality for training neural networks on participant schedules.

Key components:
- simpleLinearNet: Neural network architecture
- CreateParticipantDataset: Dataset class for loading participant data
- Training utilities: Functions for training and evaluating the network
"""
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils import basic_funcs as basic
import math
import copy
from tqdm.auto import tqdm

class simpleLinearNet(nn.Module):
    """A simple linear neural network with one hidden layer.
    
    Architecture:
    input -> hidden layer -> output
    All layers are fully connected with no bias terms.
    """
    def __init__(self, dim_input, dim_hidden, dim_output):
        super(simpleLinearNet, self).__init__()
        self.in_hid = nn.Linear(dim_input, dim_hidden, bias=False)
        self.hid_out = nn.Linear(dim_hidden, dim_output, bias=False)
        
    def forward(self, x):
        """Forward pass through the network."""
        hid = self.in_hid(x)
        out = self.hid_out(hid)
        return out, hid

def ex_initializer_(model, gamma=1e-3,mean=0.0):
    """
    In-place Re-initialization of weights

    Args:
        model: torch.nn.Module
        PyTorch neural net model
        
        gamma: float
        Initialization scale

    Returns:
        Nothing
    """
    for name, param in model.named_parameters():
        if "weight" in name:  
            n_out, n_in = param.shape
                
            if "hid_out" in name:  # Output layer weights
                std = 1e-3
            else:  # Hidden layer weights
                std = gamma
                
            nn.init.normal_(param, mean=mean, std=std)

class CreateParticipantDataset(Dataset):
    """PyTorch Dataset for participant data."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset['index'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {key: self.dataset[key][idx] for key in self.dataset}
        if self.transform:
            sample = self.transform(sample)
        return sample

def compute_accuracy(predictions, ground_truth):
    """Compute accuracy between predictions and ground truth in radians."""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    wrapped_difference = basic.wrap_to_pi(predictions - ground_truth)
    normalized_error = np.abs(wrapped_difference) / np.pi
    return 1 - normalized_error

def batch_to_torch(numpy_version):
    """Convert numpy batch to torch tensor."""
    return numpy_version.type(torch.FloatTensor)

def ordered_sweep(network, ranked_inputs):
    """Run network on ordered inputs for interpretable results."""
    preds, hids = network(ranked_inputs)
    return preds.detach().numpy().copy(), hids.detach().numpy().copy()

def run_simulation(training_params, network_params, task_parameters, df, do_test, dosave=0, sim_folder=np.nan):
    """Run neural network simulation for participant learning.
    
    1. Initializes network and loads participant data
    2. Trains network on sequence: A1 -> B -> A2
    3. Records and optionally saves results
    
    Args:
        training_params: Parameters for training (participants, epochs, etc)
        network_params: Network architecture parameters
        task_parameters: Task-specific parameters
        df: DataFrame with participant data
        do_test: Whether to run test trials
        dosave: Whether to save results
        sim_folder: Folder to save results if dosave=1
        
    Returns:
        List of results per participant
    """
    # Unpack parameters
    dim_input, dim_hidden, dim_output = network_params
    participants, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params
    
    results = []
    
    # Train network for each participant
    for idx_p, participant in tqdm(enumerate(participants)):
        print(f'Starting participant {participant}')
        
        # Get participant data
        dataset_A1, dataset_B, dataset_A2, raw_inputs, raw_labels = basic.get_datasets(df, participant, task_parameters)

        # Order inputs by feature
        A_inputs = raw_inputs[0]
        B_inputs = raw_inputs[1] 
        A_labels_feat1 = raw_labels[0, 0:2].T
        B_labels_feat1 = raw_labels[1, 0:2].T
        ordered_indices_A = basic.get_clockwise_order(A_labels_feat1)
        ordered_indices_B = basic.get_clockwise_order(B_labels_feat1)
        ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], B_inputs[ordered_indices_B]), axis=0)

        # Create data loaders
        trainloader_A1 = DataLoader(CreateParticipantDataset(dataset_A1), batch_size=batch_size, shuffle=shuffle)
        trainloader_B = DataLoader(CreateParticipantDataset(dataset_B), batch_size=batch_size, shuffle=shuffle)
        trainloader_A2 = DataLoader(CreateParticipantDataset(dataset_A2), batch_size=batch_size, shuffle=shuffle)

        # Train network through phases A1 -> B -> A2
        participant_results = runSchedule(
            train_participant_schedule, lr, gamma, n_epochs, dim_input, dim_hidden,
            dim_output, trainloader_A1, trainloader_B, trainloader_A2, ordered_inputs, do_test
        )

        participant_results['participant'] = participant
        
        # Save results if requested
        if dosave:
            file_path = f"{sim_folder}/sim_{participant}.npz"
            np.savez_compressed(file_path, **participant_results)
        
        # Cleanup
        del participant_results

    return results

def runSchedule(train_function, lr, gamma, n_epochs, dim_input, dim_hidden, dim_output, trainloader_A1, trainloader_B, trainloader_A2, ordered_inputs, do_test):
    """
    Runs a complete learning cycle:
    A: n_epochs of training on task A stimuli
    B: n_epochs of training on task B stimuli
    """
    n_train_trials = n_epochs * dim_input * 10
    n_phase = 3  # A, B, A

    # Preallocate results matrices
    results = {
        "indexes": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "inputs": np.full((n_phase, n_train_trials, dim_input), np.nan, dtype=np.float32),
        "labels": np.full((n_phase, n_train_trials, 2), np.nan, dtype=np.float32),
        "test_stim": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "probes": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "losses": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "accuracy": np.full((n_phase, n_train_trials), np.nan, dtype=np.float32),
        "predictions": np.full((n_phase, n_train_trials, dim_output), np.nan, dtype=np.float32),
        "hiddens": np.full((n_phase, n_train_trials, dim_hidden), np.nan, dtype=np.float32),
        "embeddings": np.full((n_phase, n_train_trials, dim_hidden, dim_input), np.nan, dtype=np.float32),
        "readouts": np.full((n_phase, n_train_trials, dim_output, dim_hidden), np.nan, dtype=np.float32),
    }

    # Define the network
    network = simpleLinearNet(dim_input, dim_hidden, dim_output)

    # Initialize weights
    ex_initializer_(network, gamma)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_function = nn.MSELoss()
  
    # Initial pass of the network
    initial_preds, initial_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
    results["preds_pre_training"] = initial_preds
    results["hiddens_pre_training"] = initial_hiddens

    # Training Phases
    phases = [
        (0, trainloader_A1, 1),
        (1, trainloader_B, 1),
        (2, trainloader_A2, 2),
    ]
    for phase, loader, do_update in phases:
        (
            results["indexes"][phase, :],
            results["inputs"][phase, :, :],
            results["labels"][phase, :, :],
            results["probes"][phase, :],
            results["test_stim"][phase, :],
            results["losses"][phase, :],
            results["accuracy"][phase, :],
            results["predictions"][phase, :, :],
            results["hiddens"][phase, :, :],
            results["embeddings"][phase, :, :, :],
            results["readouts"][phase, :, :, :],
        ) = train_function(
            network, loader, n_epochs, loss_function, optimizer, do_update, do_test
        )

        # Post-phase ordered sweep
        post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
        results[f"preds_post_phase_{phase}"] = post_preds
        results[f"hiddens_post_phase_{phase}"] = post_hiddens

    return results





def train_participant_schedule(network, trainloader, n_epochs, loss_function, optimizer, do_update, do_test):
    """
    Train the network on x-y coordinates 

    Args:
        network: The neural network to be trained.
        trainloader: DataLoader object containing training data.
        n_epochs: Number of epochs to train.
        loss_function: The loss function to use.
        optimizer: Optimizer for updating network parameters.
        do_update: Controls how updates are applied (0 = no update, 1 = standard, 2 = conditional on feature_probe).

    Returns:
        tuple: Various metrics including indexes, inputs, labels, probes, losses, accuracy, predictions, hiddens, embeddings, readouts.
    """
    # Initialize storage lists
    metrics = {
        "indexes": [],
        "losses": [],
        "accuracy": [],
        "predictions": [],
        "hiddens": [],
        "embeddings": [],
        "readouts": [],
        "probes": [],
        "test_stim":[],
        "labels": [],
        "inputs": [],
    }

    for epoch in range(n_epochs):
        for batch_idx, data in enumerate(trainloader):
            # Reset gradients
            optimizer.zero_grad()

            # Extract batch data
            index = data['stim_index']
            input = batch_to_torch(data['input'])
            label_x = batch_to_torch(data['label_x'])
            label_y = batch_to_torch(data['label_y'])
            feature_probe = batch_to_torch(data['feature_probe'])
            test_stim = batch_to_torch(data['test_stim'])
            
                    
            joined_label = torch.cat((label_x.unsqueeze(1), label_y.unsqueeze(1)), dim=1)
            radians_label = math.atan2(label_x, label_y)

            # Forward pass
            out, hid = network(input)

            # Calculate loss based on feature probe
            if feature_probe == 0:
                loss = loss_function(out[:, :2], joined_label)
                pred_rads = math.atan2(out[:, 0].detach().numpy(),out[:, 1].detach().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label)
                
            elif feature_probe == 1:
                loss = loss_function(out[:, 2:4], joined_label)
                pred_rads = math.atan2(out[:, 2].detach().numpy(),out[:, 3].detach().numpy())
                accuracy = compute_accuracy(pred_rads, radians_label)
                
            else:
                raise ValueError("Undefined loss setting for feature_probe.")

            # Update network if required
            if do_update == 1 and do_test==1 and test_stim.numpy() == 0:
              loss.backward()
              optimizer.step()
            elif do_update == 1 and do_test ==0:
              loss.backward()
              optimizer.step()
            elif do_update == 2 and feature_probe == 0:  # In C, only update for feature 0 
              loss.backward()
              optimizer.step()

            # Store metrics
            metrics["indexes"].append(index)
            metrics["inputs"].append(input.numpy())
            metrics["labels"].append(joined_label.numpy())
            metrics["probes"].append(feature_probe.numpy())
            metrics["test_stim"].append(test_stim.numpy())
            metrics["losses"].append(loss.item())
            metrics["accuracy"].append(accuracy)
            metrics["predictions"].append(np.expand_dims(out.detach().numpy(), axis=1))
            metrics["hiddens"].append(hid.detach().numpy())
            metrics["embeddings"].append(network.in_hid.weight.detach().numpy())
            metrics["readouts"].append(network.hid_out.weight.detach().numpy())

    # Convert lists to arrays where applicable
    metrics = {key: np.squeeze(value) for key, value in metrics.items()}
    
    return (
        metrics["indexes"],
        metrics["inputs"],
        metrics["labels"],
        metrics["probes"],
        metrics["test_stim"],
        metrics["losses"],
        metrics["accuracy"],
        metrics["predictions"],
        metrics["hiddens"],
        metrics["embeddings"],
        metrics["readouts"],
    )


def train_single_schedule(training_params, network_params, task_parameters, df, do_test):
    
    dim_input, dim_hidden, dim_output = network_params
    _, n_phase, n_epochs, n_train_trials, shuffle, batch_size, gamma, lr = training_params
    
    # Phantom df created where all groups trained on same A, for geometry visualisation
    dataset_A1, dataset_B_same, dataset_A2, raw_inputs, raw_labels = basic.get_datasets(df, 'geom_sub_same', task_parameters)
    _, dataset_B_near, _, _, _ = basic.get_datasets(df, 'geom_sub_near', task_parameters)
    _, dataset_B_far, _, _, _ = basic.get_datasets(df, 'geom_sub_far', task_parameters)
    
    A_inputs = raw_inputs[0]
    B_inputs = raw_inputs[1]
    A_labels_feat1 = raw_labels[0, 0:2].T
    B_labels_feat1 = raw_labels[1, 0:2].T

    ordered_indices_A = basic.get_clockwise_order(A_labels_feat1)
    ordered_indices_B = basic.get_clockwise_order(B_labels_feat1)
    ordered_inputs = np.concatenate((A_inputs[ordered_indices_A], B_inputs[ordered_indices_B]), axis=0)

    trainloader_A1 = DataLoader(CreateParticipantDataset(dataset_A1), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_same = DataLoader(CreateParticipantDataset(dataset_B_same), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_near = DataLoader(CreateParticipantDataset(dataset_B_near), batch_size=batch_size, shuffle=shuffle)
    trainloader_B_far = DataLoader(CreateParticipantDataset(dataset_B_far), batch_size=batch_size, shuffle=shuffle)
    trainloader_A2 = DataLoader(CreateParticipantDataset(dataset_A2), batch_size=batch_size, shuffle=shuffle)

    
    n_train_trials = n_epochs * dim_input * 10
    n_phase = 3  # A, B, A

    # Preallocate results matrices

    results = {
        "indexes": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "inputs": np.full((3, n_phase, n_train_trials, dim_input), np.nan, dtype=np.float32),
        "labels": np.full((3, n_phase, n_train_trials, 2), np.nan, dtype=np.float32),
        "test_stim": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "probes": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "losses": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "accuracy": np.full((3, n_phase, n_train_trials), np.nan, dtype=np.float32),
        "predictions": np.full((3, n_phase, n_train_trials, dim_output), np.nan, dtype=np.float32),
        "hiddens": np.full((3, n_phase, n_train_trials, dim_hidden), np.nan, dtype=np.float32),
        "embeddings": np.full((3, n_phase, n_train_trials, dim_hidden, dim_input), np.nan, dtype=np.float32),
        "readouts": np.full((3, n_phase, n_train_trials, dim_output, dim_hidden), np.nan, dtype=np.float32),
        "preds_pre_training": np.full((3, task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_pre_training": np.full((3, task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_0": np.full((3,  task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_0": np.full((3,  task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_1": np.full((3, task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_1": np.full((3,  task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),
        "preds_post_phase_2": np.full((3,  task_parameters['nStim_perTask']*2, dim_output), np.nan, dtype=np.float32),
        "hiddens_post_phase_2": np.full((3,  task_parameters['nStim_perTask']*2, dim_hidden), np.nan, dtype=np.float32),

    }

    # Define the network
    network = simpleLinearNet(dim_input, dim_hidden, dim_output)

    # Initialize weights
    ex_initializer_(network, gamma)

    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    loss_function = nn.MSELoss()
  
    # Initial pass of the network
    initial_preds, initial_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
    results["preds_pre_training"] = initial_preds
    results["hiddens_pre_training"] = initial_hiddens
    
    results["indexes"][0, 0, :], results["inputs"][0,0, :, :],results["labels"][0,0, :, :],results["probes"][0,0, :],results["test_stim"][0,0, :],results["losses"][0,0, :],results["accuracy"][0,0, :],results["predictions"][0,0, :, :],results["hiddens"][0,0, :, :],results["embeddings"][0,0, :, :, :],results["readouts"][0,0, :, :, :] = train_participant_schedule(network, trainloader_A1, n_epochs, loss_function, optimizer, 1, do_test)
    
    # Post-phase ordered sweep
    post_preds, post_hiddens = ordered_sweep(network, torch.from_numpy(ordered_inputs).float())
    results[f"preds_post_phase_0"][0,:,:] = post_preds
    results[f"hiddens_post_phase_0"][0,:,:] = post_hiddens
     
    # Now split the network into the three sessions 
    network_same = copy.deepcopy(network)
    network_near = copy.deepcopy(network)
    network_far = copy.deepcopy(network)
    optimizer_same = torch.optim.SGD(network_same.parameters(), lr=lr)
    optimizer_near = torch.optim.SGD(network_near.parameters(), lr=lr)
    optimizer_far = torch.optim.SGD(network_far.parameters(), lr=lr)
        
    for condition_idx, (condition_network, condition_training, condition_optimizer) in enumerate(zip(
    [network_same, network_near, network_far],
    [trainloader_B_same, trainloader_B_near, trainloader_B_far],
    [optimizer_same, optimizer_near, optimizer_far])): 
         
        phases = [
            (1, condition_training, 1),
            (2, trainloader_A2, 2),
        ]
         
        for phase, loader, do_update in phases:
            
            (results["indexes"][condition_idx, phase, :],
            results["inputs"][condition_idx, phase, :, :],
            results["labels"][condition_idx, phase, :, :],
            results["probes"][condition_idx, phase, :],
            results["test_stim"][condition_idx, phase, :],
            results["losses"][condition_idx, phase, :],
            results["accuracy"][condition_idx, phase, :],
            results["predictions"][condition_idx, phase, :, :],
            results["hiddens"][condition_idx, phase, :, :],
            results["embeddings"][condition_idx, phase, :, :, :],
            results["readouts"][condition_idx, phase, :, :, :],
            ) = train_participant_schedule(condition_network, loader, n_epochs, loss_function, condition_optimizer, do_update, do_test)

            # Post-phase ordered sweep
            post_preds, post_hiddens = ordered_sweep(condition_network, torch.from_numpy(ordered_inputs).float())
            results[f"preds_post_phase_{phase}"][condition_idx,:,:] = post_preds
            results[f"hiddens_post_phase_{phase}"][condition_idx,:,:] = post_hiddens
            
            
   
    return results