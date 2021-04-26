"""
Functions for saving results
"""

import numpy as np
import pandas as pd
import torch

from networks import ZipperNN

def save_performance(directory, file_prefix, prev_network, test_dataset):
    """
    Save the classifications, network monitoring, network, and test_dataset
    
    Args:
        directory (str): directory name to put output, no trailing '/'
        file_prefix (str): prefix for outfiles
        network: A trained neural network
        test_dataset: the dataset used for testing
    """
    # Load best network
    network = ZipperNN(4, 4, 4)
    if file_prefix.endswith('_train'):
        net_name = file_prefix[:-6]
    else:
        net_name = file_prefix
    network.load_state_dict(torch.load(f"{directory}/{net_name}_network.pt"))
    network.eval()
    
    # Save classifications
    labels = test_dataset[:]['label'].data.numpy()
    res = network(test_dataset[:]['lightcurve'], test_dataset[:]['image']).detach().numpy()
    output = np.hstack((res, labels.reshape(len(labels), 1)))
    columns = ["No Lens", "Lens", "LSNIa", "LSNCC", "Label"]
    df = pd.DataFrame(data=output, columns=columns)
    df.to_csv(f"{directory}/{file_prefix}_classifications.csv", index=False)
    
    # Save network performance
    out_data = [(a, b, c) for a, b, c in zip(prev_network.losses, prev_network.train_acc, prev_network.validation_acc)]
    out_columns = ["Loss", "Train Acc", "Test Acc"]
    df = pd.DataFrame(data=out_data, columns=out_columns)
    df.to_csv(f"{directory}/{file_prefix}_monitoring.csv", index=False)
    
    # Save dataset
    torch.save(test_dataset, f"{directory}/{file_prefix}_dataset.pt")
