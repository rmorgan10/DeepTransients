"""
Functions for saving results
"""

import numpy as np
import pandas as pd
import torch

from networks import ZipperNN, CNN_single, RNN_single

def save_performance(directory, file_prefix, net_type, prev_network, test_dataset, train=False):
    """
    Save the classifications, network monitoring, network, and test_dataset
    
    Args:
        directory (str): directory name to put output, no trailing '/'
        file_prefix (str): prefix for outfiles
        net_type (str): type of netowrk like RNN, CNN, or ZIPPER
        network: A trained neural network
        test_dataset: the dataset used for testing
        train (bool, default=False): save the training performance
    """
    # Handle train flag
    train_info = "_train" if train else ""
    
    # Load best network
    networks = {'ZIPPER': ZipperNN(4, 4, 4),
            'CNN': CNN_single(4, 2),
            'RNN': RNN_single(4, 3)}
    network = networks[net_type]
    
    network.load_state_dict(torch.load(f"{directory}/{file_prefix}_{net_type}_network.pt"))
    network.eval()
    
    # Save classifications
    labels = test_dataset[:]['label'].data.numpy()
    if net_type == 'CNN':
        res = network(test_dataset[:]['image']).detach().numpy()
        columns = ["No Lens", "Lens", "Label"]
    elif net_type == 'RNN':
        res = network(test_dataset[:]['lightcurve']).detach().numpy()
        columns = ["No LSN", "LSNIa", "LSNCC", "Label"]
    elif net_type == 'ZIPPER':
        res = network(test_dataset[:]['lightcurve'], test_dataset[:]['image']).detach().numpy()
        columns = ["No Lens", "Lens", "LSNIa", "LSNCC", "Label"]
    else:
        raise NotImplementedError("Net type must be RNN, CNN, or ZIPPER")

    output = np.hstack((res, labels.reshape(len(labels), 1)))
    df = pd.DataFrame(data=output, columns=columns)
    df.to_csv(f"{directory}/{file_prefix}_{net_type}_classifications{train_info}.csv", index=False)
    
    # Save network performance
    out_data = [(a, b, c) for a, b, c in zip(prev_network.losses, prev_network.train_acc, prev_network.validation_acc)]
    out_columns = ["Loss", "Train Acc", "Test Acc"]
    df = pd.DataFrame(data=out_data, columns=out_columns)
    df.to_csv(f"{directory}/{file_prefix}_{net_type}_monitoring.csv", index=False)
    
    # Save dataset
    torch.save(test_dataset, f"{directory}/{file_prefix}_{net_type}_dataset{train_info}.pt")
