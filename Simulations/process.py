"""
Process deeplenstronomy simulations for ZipperNet
"""

import glob

import numpy as np
import pandas as pd
from scipy import ndimage


def filter_nans(images, metadata, band='g'):
    """
    Remove examples where any image in the time series (in 
    any band) contains NaNs. Prints fraction of data removed.
    
    Args:
        images (np.array): shape (N, <num_bands>, <height>, <width>)
        metadata (pd.DataFrame): length N dataframe of metadata
        band (str, default='g'): band to use for metadata
        
    Returns:
        images where a NaN wasn't present, 
        metadata where a NaN wasn't present
    """
    # Find the OBJIDs of the time series examples with NaNs
    mask = (np.sum(np.isnan(images), axis=(-1, -2, -3)) > 0)
    bad_objids = metadata[f'OBJID-{band}'].values[mask]
    full_mask = np.array([x in bad_objids for x in metadata[f'OBJID-{band}'].values])
    
    # Determine the data loss
    print("losing", round(sum(full_mask) / len(images) * 100, 2), "%")
    
    # Apply the mask and return
    return images[~full_mask], metadata[~full_mask].copy().reset_index(drop=True)


def coadd_bands(image_arr):
    """
    Average an array of images in each band
    
    Args:
        image_arr (np.array): shape (N, <num_bands>, <height>, <width>)
        
    Returns:
        coadded array with shape (<num_bands>, <height>, <width>)
    """
    return np.mean(image_arr, axis=0)
    
def scale_bands(coadded_image_arr):
    """
    Scale pixel values to 0 to 1 preserving color
    
    Args:
        coadded_image_arr (np.array): shape (<num_bands>, <height>, <width>)
        
    Returns:
        scaled array with shape (<num_bands>, <height>, <width>)
    """
    return (coadded_image_arr - coadded_image_arr.min()) / (coadded_image_arr - coadded_image_arr.min()).max()


def extract_lightcurves(images, aperture_rad=20):
    """
    Measure pixel values for each band
    
    Args:
        images (np.array): one time-series example shape (m, <num_bands>, <height>, <width>)
        aperture_rad (int, default=20): radius in pixels of the aperture to use
    
    Returns:
        lightcurve array for the example
    """
    # construct aperature mask
    yy, xx = np.meshgrid(range(np.shape(images)[-1]), range(np.shape(images)[-1]))
    center = int(round(np.shape(images)[-1] / 2))
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    aperature = (dist <= 20)
    
    # make time measurements
    sum_in_aperature = np.sum(images[:,:,aperature], axis=-1)
    med_outside_aperature = np.median(images[:,:,~aperature], axis=-1)
    res = sum_in_aperature - med_outside_aperature * aperature.sum()
    
    return res

def process(image_arr, metadata, band='g'):
    """
    Iterate through image_arr and process data
    
    Args:
        image_arr (np.array): shape (N, <num_bands>, <height>, <width>)
        metadata (pd.DataFrame): length N dataframe of metadata
        band (str, default='g'): band to use for metadata
        
    Returns:
        processed_ims with shape (N, <num_bands>, <height>, <width>),
        lightcurves
    """
    # Clean the data
    clean_ims, clean_md = filter_nans(image_arr, metadata)
    
    # Separate by cadence length
    outdata = {}
    
    # Iterate through data
    out_ims, out_lcs = [], []
    current_objid = clean_md[f'OBJID-{band}'].values.min()
    prev_idx = 0
    for idx, objid in enumerate(clean_md[f'OBJID-{band}'].values):
        
        if objid != current_objid:
            
            # Select the object
            example = clean_ims[prev_idx:idx,:,:,:]
            
            # Select the metadata
            example_md = clean_md.loc[prev_idx:idx]
            
            # Determine cadence length
            key = len(example)
            if key not in outdata:
                outdata[key] = {"ims": [], 'lcs': [], 'mds': []}
            
            # Coadd and scale the images
            processed_ims = scale_bands(coadd_bands(example))
            
            # Measure and scale the lightcurves
            processed_lcs = scale_bands(extract_lightcurves(example))

            # Append to output
            outdata[key]["ims"].append(processed_ims)
            outdata[key]["lcs"].append(processed_lcs)
            outdata[key]["mds"].append(example_md)

            # Update trackers
            prev_idx = idx
            current_objid = objid
            
    return outdata

def mirror_and_rotate(data):
    """
    Apply a complete set of 2D mirrorings and rotations

    Args:
        data (dict): output of process()

    Returns:
        outdata (dict): Same as data, but has mirrored and rotated copies appended
    """

    outdata = {}
    for key in data.keys():
        outdata[key] = {'ims': [], 'lcs': [], 'mds': []}
        
        # Rotate and mirror the images, duplicate the metadata and lightcurves
        for angle in [0.0, 90.0, 180.0, 270.0]:
            rotated_ims = ndimage.rotate(data[key]['ims'], axes=(-1,-2), angle=angle, reshape=False)

            # Append rotated images to output
            outdata[key]["ims"].append(rotated_ims)
            outdata[key]["lcs"].append(data[key]['lcs'])
            outdata[key]["mds"].extend(data[key]['mds'])

            # Mirror images and append to output
            outdata[key]["ims"].append(rotated_ims[:,:,::-1,:])
            outdata[key]["lcs"].append(data[key]['lcs'])
            outdata[key]["mds"].extend(data[key]['mds'])

        # Stack results
        outdata[key]["ims"] = np.concatenate(outdata[key]["ims"])
        outdata[key]["lcs"] = np.concatenate(outdata[key]["lcs"])
            
    return outdata
            


def run(directory, configuration, show=True):
    print("Processing ", configuration)
    
    # Ingest
    images = np.load(f'{directory}/{configuration}_images.npy')
    metadata = pd.read_csv(f'{directory}/{configuration}_metadata.csv')
    
    # Process
    outdata = process(images, metadata, band='g')

    # Mirror and Rotate
    outdata = mirror_and_rotate(outdata)
    
    # Create arrays for each cadence length and save
    for key in outdata:
        
        out_ims = np.array(outdata[key]["ims"])
        out_lcs = np.array(outdata[key]["lcs"])
        out_md = {idx: outdata[key]["mds"][idx] for idx in range(len(outdata[key]["mds"]))}
        
        np.save(f"{directory}/{configuration}_proc_ims_{key}.npy", out_ims)
        np.save(f"{directory}/{configuration}_proc_lcs_{key}.npy", out_lcs)
        np.save(f"{directory}/{configuration}_proc_mds_{key}.npy", out_md, allow_pickle=True)

if __name__ == '__main__':
    import utils

    directory = utils.get_dataset_name()

    configurations = [x.split("/")[1].split("_images.")[0] for x in glob.glob(f"{directory}/*_images.npy")]
    for configuration in sorted(configurations):
        run(directory, configuration)
