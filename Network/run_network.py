"""
Run ZipperNet on a dataset
"""

import data_utils
import networks
import save
import training
import utils

# Get dataset name
dataset_name = utils.get_dataset_name()
suffix = utils.get_suffix()

# Instantiate a ZipperNet
zipper = networks.ZipperNN(4, 4, 4)

# Ingest the data
groups = ['GROUP_1', 'GROUP_2', 'GROUP_3', 'GROUP_4']
print("Loading data")
train_dataset, test_dataset = data_utils.make_train_test_datasets(dataset_name, groups, suffix)

# Get dataloader
train_dataloader = data_utils.make_dataloader(train_dataset)

print("Training ZipperNet")
# Train ZipperNet
zipper = training.train_zipper(zipper,
                               train_dataloader,
                               train_dataset,
                               test_dataset,
                               monitor=True,
                               outfile_prefix=f"{dataset_name}/{dataset_name}")

print("Saving results")
# Save the performance
save.save_performance(dataset_name, dataset_name, zipper, test_dataset)
save.save_performance(dataset_name, dataset_name + '_train', zipper, train_dataset)
