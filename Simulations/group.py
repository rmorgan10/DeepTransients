"""
Group configurations for ZipperNet
"""
import glob

import numpy as np


def group_data(directory: str, group_dict: dict, suffix: int):
    for group, info in group_dict.items():
        print(f"{group} -- SUFFIX {suffix}") 
        images, metadata, lightcurves = [], [], []
        
        # Ingest
        for configuration in info['CONFIGURATIONS']:
            images.append(np.load(f"{directory}/{configuration}_proc_ims_{suffix}.npy"))
            lightcurves.append(np.load(f"{directory}/{configuration}_proc_lcs_{suffix}.npy"))
            metadata.append(np.load(f"{directory}/{configuration}_proc_mds_{suffix}.npy", allow_pickle=True).item())
            
        # Combine
        out_ims = np.concatenate(images)
        out_lcs = np.concatenate(lightcurves)
        out_mds = {}
        start = 0
        for md in metadata:
            for k, v in md.items():
                out_mds[start + k] = v
            start += len(md)
        
        # Output
        np.save(f"{directory}/{group}_ims_{suffix}.npy", out_ims)
        np.save(f"{directory}/{group}_lcs_{suffix}.npy", out_lcs)
        np.save(f"{directory}/{group}_mds_{suffix}.npy", out_mds, allow_pickle=True)


if __name__ == '__main__':
    import utils
    
    # Establish groups
    group_dict = {'GROUP_1': {'NAME': "No Lensing",
                              'CONFIGURATIONS': ['CONFIGURATION_2', 
                                                 'CONFIGURATION_5', 
                                                 'CONFIGURATION_8', 
                                                 'CONFIGURATION_11', 
                                                 'CONFIGURATION_13', 
                                                 'CONFIGURATION_14', 
                                                 'CONFIGURATION_15',
                                                 'CONFIGURATION_16',
                                                 'CONFIGURATION_17']},
                  'GROUP_2': {'NAME': "Lensing",
                              'CONFIGURATIONS': ['CONFIGURATION_1', 
                                                 'CONFIGURATION_6', 
                                                 'CONFIGURATION_7', 
                                                 'CONFIGURATION_12']},
                  'GROUP_3': {'NAME': "LSNE-Ia",
                              'CONFIGURATIONS': ['CONFIGURATION_3', 
                                                 'CONFIGURATION_9']},
                  'GROUP_4': {'NAME': "LSNE-CC",
                              'CONFIGURATIONS': ['CONFIGURATION_4', 
                                                 'CONFIGURATION_10']}}

    # Get dataset name
    dataset_name = utils.get_dataset_name()

    # Find cadence suffixes
    names = glob.glob(f"{dataset_name}/CONFIGURATION_1_proc_ims_*.npy")
    suffixes = [int(x.split('proc_ims_')[-1].split('.npy')[0]) for x in names]

    # Group configurations
    for suffix in suffixes:
        group_data(dataset_name, group_dict, suffix) 
