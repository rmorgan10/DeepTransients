"""
Run ZipperNet on a dataset
"""
import sys

import data_utils
import networks
import save
import training
import utils

# Get dataset name
dataset_name = utils.get_dataset_name()
suffix = utils.get_suffix()

# Get network type
if 'CNN' in [x.upper() for x in sys.argv]:
    net_type = 'CNN'
elif 'RNN' in [x.upper() for x in sys.argv]:
    net_type = 'RNN'
else:
    net_type = 'ZIPPER'
print(f"Running {net_type}")


# Instantiate a network
network_types = {'ZIPPER': networks.ZipperNN(4, 4, 4),
                 'CNN': networks.CNN_single(4, 2),
                 'RNN': networks.RNN_single(4, 3)}
network = network_types[net_type]

# Ingest the data
label_maps = {'ZIPPER': {},
              'CNN': {'GROUP_1': 0, 'GROUP_2': 1, 'GROUP_3': 1, 'GROUP_4': 1},
              'RNN': {'GROUP_1': 0, 'GROUP_2': 0, 'GROUP_3': 1, 'GROUP_4': 2}}
groups = ['GROUP_1', 'GROUP_2', 'GROUP_3', 'GROUP_4']
print("Loading data")
train_dataset, test_dataset = data_utils.make_train_test_datasets(dataset_name, groups, suffix, label_map=label_maps[net_type])

# Get dataloader
train_dataloader = data_utils.make_dataloader(train_dataset)

# Train network
print(f"Training {net_type}")
if net_type == 'ZIPPER':
    network = training.train_zipper(network,
                                    train_dataloader,
                                    train_dataset,
                                    test_dataset,
                                    monitor=True,
                                    outfile_prefix=f"{dataset_name}/{dataset_name}_{net_type}")
elif net_type == 'RNN':
    network = training.train_single(network,
                                    train_dataloader,
                                    train_dataset,
                                    test_dataset,
                                    'lightcurve',
			            monitor=True,
                                    outfile_prefix=f"{dataset_name}/{dataset_name}_{net_type}")
elif net_type == 'CNN':
    network = training.train_single(network,
                                    train_dataloader,
                                    train_dataset,
                                    test_dataset,
                                    'image',
                                    monitor=True,
                                    outfile_prefix=f"{dataset_name}/{dataset_name}_{net_type}")
else:
    raise NotImplementedError(f"Unexpected control flow caused by net_type = {net_type}")

print("Saving results")
# Save the performance
save.save_performance(dataset_name, dataset_name, net_type, network, test_dataset)
save.save_performance(dataset_name, dataset_name, net_type, network, train_dataset, train=True)
