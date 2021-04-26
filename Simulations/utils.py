"""
Helper functions
"""

import os
import sys

def get_dataset_name():
    """
    Collect dataset name from command line args

    Returns:
        dataset_name as string if command line arg is valid

    Raises:
        ValueError if passed dataset name is not valid
        KeyError if no dataset name is passed
    """

    # Set allowed dataset names
    datasets = ['full_data', 'high_cad_data', 'lsst_data', 'des_deep_data', 'test_data']

    # Get dataset name
    try:
        dataset_name = sys.argv[1]
    except IndexError:
        raise KeyError("No dataset name was passed as a command line arg")

    # Check that the name is valid
    force = "--force" in sys.argv
    if dataset_name not in datasets:
        if not force:
            raise ValueError(f"{dataset_name} is not a valid dataset")

    return dataset_name


def get_suffix():
    # Get dataset name
    try:
        suffix = sys.argv[2]
    except IndexError:
        raise KeyError("No suffix name was passed as a command line arg")

    # Check that suffix is valid
    dataset_name = get_dataset_name()
    if not os.path.exists(f'{dataset_name}/GROUP_1_ims_{suffix}.npy'):
        raise ValueError(f"{suffix} is not a valid suffix for {dataset_name}")

    return int(suffix)
