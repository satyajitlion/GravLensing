# In normalize_test_data.py
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_test(dataset, max_images, scalers):
    """
    Normalize test data using pre-fitted scalers from training data
    
    Parameters:
    - dataset: Your test dataset (SIE, SIE+shear, etc.)
    - max_images: Maximum number of images (same as used in training)
    - scalers: The scalers dictionary returned from extract_lens_data()
    
    Returns:
    - x_test: Normalized input features
    - y_test: Normalized output targets
    """
    inputs = []
    outputs = []
    
    # Extract the scalers
    img_scaler = scalers['img_scaler']
    time_scaler = scalers['time_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']
    
    for system in dataset:
        # Normalize each component using the EXISTING scalers
        img_normalized = img_scaler.transform(system['img'])
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_2d = time_1d.reshape(-1, 1)
        time_normalized = time_scaler.transform(time_2d).flatten()
        
        # Handle redshifts
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_2d = potent_1d.reshape(-1, 1)
        potent_normalized = potent_scaler.transform(potent_2d).flatten()
        
        # Handle deflections
        deflec_normalized = deflec_scaler.transform(system['deflec'])
        
        # Pad to max_images (same as training)
        img_padded = np.zeros(max_images * 2)
        img_flat = img_normalized.flatten()
        img_padded[:len(img_flat)] = img_flat
        
        time_padded = np.zeros(max_images)
        time_padded[:len(time_normalized)] = time_normalized
        
        potent_padded = np.zeros(max_images)
        potent_padded[:len(potent_normalized)] = potent_normalized
        
        deflec_padded = np.zeros(max_images * 2)
        deflec_flat = deflec_normalized.flatten()
        deflec_padded[:len(deflec_flat)] = deflec_flat
        
        # Combine inputs and outputs
        input_vec = np.concatenate([img_padded, time_padded, z_normalized])
        output_vec = np.concatenate([potent_padded, deflec_padded])
        
        inputs.append(input_vec)
        outputs.append(output_vec)
    
    return np.array(inputs), np.array(outputs)