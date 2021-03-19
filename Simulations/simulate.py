"""
Trigger a deeplenstronomy simulation
"""

from deeplenstronomy import deeplenstronomy as dl

import utils

# Get the dataset name
dataset_name = utils.get_dataset_name()

# Generate the dataset
dataset = dl.make_dataset(f"{dataset_name}.yaml", 
                          store_in_memory=False, 
                          save_to_disk=True, 
                          verbose=True, 
                          solve_lens_equation=False)

