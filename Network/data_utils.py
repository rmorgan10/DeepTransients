"""
Dataset and Dataloader objects for DeepTransients
"""

import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data import TensorDataset

class CombinedDataset(Dataset):
    """Dataset of DeepLenstronomy Lightcurves and Images"""

    def __init__(self, images, lightcurves, labels, transform=None):
        """
        docstring
        """
        self.images = images
        self.lightcurves = lightcurves
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        lightcurve = self.lightcurves[idx]
        label = np.array(self.labels[idx])
        
        sample = {'lightcurve': lightcurve, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToCombinedTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        lightcurve, image, label = sample['lightcurve'], sample['image'], sample['label']

        return {'lightcurve': torch.from_numpy(lightcurve).float(),
                'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label)}



def make_train_test_datasets(directory: str, class_names: list, suffix: int, transform=ToCombinedTensor(), label_map={}):
    images, lightcurves, labels, metadata = [], [], [], []

    # Ingest and label data
    for label, class_name in enumerate(class_names):
        # Lightcurves
        lc_file = f'{directory}/{class_name}_lcs_{suffix}.npy'
        lc = np.load(lc_file)
        print(lc.shape)
        lightcurves.append(lc)
        
        # Images
        im_file = f'{directory}/{class_name}_ims_{suffix}.npy'
        im = np.load(im_file)
        images.append(im.reshape((len(im), 4, 45, 45)))
        
        # Labels
        class_label = label_map[class_name] if class_name in label_map else label
        print(class_label)
        labels.extend([class_label]*len(im))
        
        # Metadata
        md_file = f'{directory}/{class_name}_mds_{suffix}.npy'
        metadata.append(np.load(md_file, allow_pickle=True).item())

    # Shuffle and split data
    X_lc = np.concatenate(lightcurves)
    X_im = np.concatenate(images)
    y = np.array(labels, dtype=int)
    train_lightcurves, test_lightcurves, train_labels, test_labels = train_test_split(
        X_lc, y, test_size=0.1, random_state=6, stratify=y)
    
    train_images, test_images, garb1, garb2 = train_test_split(
        X_im, y, test_size=0.1, random_state=6, stratify=y)
    
    # Split and save metadata
    full_md = []
    for md in metadata:
        for v in md.values():
            full_md.append(v)
    
    train_md, test_md, garb1, garb2 = train_test_split(
        full_md, y, test_size=0.1, random_state=6, stratify=y)
    
    train_md = {idx: train_md[idx] for idx in range(len(train_md))}
    test_md = {idx: test_md[idx] for idx in range(len(test_md))}
    
    np.save(f"{directory}/{directory}_train_md_{suffix}.npy", train_md, allow_pickle=True)
    np.save(f"{directory}/{directory}_test_md_{suffix}.npy", test_md, allow_pickle=True)
    
    
    # Create a PyTorch Dataset
    return (CombinedDataset(train_images, train_lightcurves, train_labels, transform=transform),
            CombinedDataset(test_images, test_lightcurves, test_labels, transform=transform))
        

def make_dataloader(dataset, batch_size=5, shuffle=True, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

