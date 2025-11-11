import numpy as np
import glob

folder_path = "AmarelOutput"

def combine_arrays(array_type, folder=folder_path):
    file_pattern = f"{folder}/{array_type}_*.npy"
    files = sorted(glob.glob(file_pattern))
    arrays = [np.load(file, allow_pickle=True) for file in files]
    return np.concatenate(arrays, axis=0)

# Combine all arrays
valShear = combine_arrays('valShear', folder_path)
valEllip = combine_arrays('valEllip', folder_path)
valBoth = combine_arrays('valBoth', folder_path)

# Saving the arrays

output_folder = "CombinedArrays"

np.save(f"{output_folder}/valShear.npy", valShear)
np.save(f"{output_folder}/valEllip.npy", valEllip)
np.save(f"{output_folder}/valBoth.npy", valBoth)