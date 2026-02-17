import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_test_singles(dataset, scalers):
    """
    Normalize test data for single image lenses using pre-fitted scalers from training data
    
    Parameters:
    - dataset: Your test dataset (SIE, SIE+shear, etc.) with only single image lenses
    - scalers: The scalers dictionary returned from extract_single_lens_data()
    
    Returns:
    - x_test: Normalized input features for singles
    - y_test: Normalized output targets for singles
    """
    inputs = []
    outputs = []
    
    # Filter for systems with exactly 1 image
    single_systems = [s for s in dataset if s['img'].shape[0] == 1]
    
    # Extract the scalers
    img_scaler = scalers['img_scaler']
    time_scaler = scalers['time_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']
    
    for system in single_systems:
        # Normalize each component using the EXISTING scalers
        img_normalized = img_scaler.transform(system['img']).flatten()  # Shape: (2,)
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (1,)
        
        # Handle redshifts
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (1,)
        
        # Handle deflections
        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()  # Shape: (2,)
        
        # Combine inputs and outputs (NO PADDING!)
        # Input: 2 (img) + 1 (time) + 2 (z) = 5 features
        input_vec = np.concatenate([
            img_normalized,
            time_normalized,
            z_normalized
        ])
        
        # Output: 1 (potent) + 2 (deflec) = 3 features
        output_vec = np.concatenate([
            potent_normalized,
            deflec_normalized
        ])
        
        inputs.append(input_vec)
        outputs.append(output_vec)
    
    return np.array(inputs), np.array(outputs)

def normalize_test_doubles(dataset, scalers):
    """
    Normalize test data for double image lenses using pre-fitted scalers from training data
    
    Parameters:
    - dataset: Your test dataset (SIE, SIE+shear, etc.) with only double image lenses
    - scalers: The scalers dictionary returned from extract_double_lens_data()
    
    Returns:
    - x_test: Normalized input features for doubles
    - y_test: Normalized output targets for doubles
    """
    inputs = []
    outputs = []
    
    # Filter for systems with exactly 2 images
    double_systems = [s for s in dataset if s['img'].shape[0] == 2]
    
    # Extract the scalers
    img_scaler = scalers['img_scaler']
    time_scaler = scalers['time_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']
    
    for system in double_systems:
        # Normalize each component using the EXISTING scalers
        img_normalized = img_scaler.transform(system['img']).flatten()  # Shape: (4,)
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (2,)
        
        # Handle redshifts
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (2,)
        
        # Handle deflections
        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()  # Shape: (4,)
        
        # Combine inputs and outputs (NO PADDING!)
        # Input: 4 (img) + 2 (time) + 2 (z) = 8 features
        input_vec = np.concatenate([
            img_normalized,
            time_normalized,
            z_normalized
        ])
        
        # Output: 2 (potent) + 4 (deflec) = 6 features
        output_vec = np.concatenate([
            potent_normalized,
            deflec_normalized
        ])
        
        inputs.append(input_vec)
        outputs.append(output_vec)
    
    return np.array(inputs), np.array(outputs)

def normalize_test_quads(dataset, scalers):
    """
    Normalize test data for quad image lenses using pre-fitted scalers from training data
    
    Parameters:
    - dataset: Your test dataset (SIE, SIE+shear, etc.) with only quad image lenses
    - scalers: The scalers dictionary returned from extract_quad_lens_data()
    
    Returns:
    - x_test: Normalized input features for quads
    - y_test: Normalized output targets for quads
    """
    inputs = []
    outputs = []
    
    # Filter for systems with exactly 4 images
    quad_systems = [s for s in dataset if s['img'].shape[0] == 4]
    
    # Extract the scalers
    img_scaler = scalers['img_scaler']
    time_scaler = scalers['time_scaler']
    z_scaler = scalers['z_scaler']
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']
    
    for system in quad_systems:
        # Normalize each component using the EXISTING scalers
        img_normalized = img_scaler.transform(system['img']).flatten()  # Shape: (8,)
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (4,)
        
        # Handle redshifts
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (4,)
        
        # Handle deflections
        deflec_normalized = deflec_scaler.transform(system['deflec']).flatten()  # Shape: (8,)
        
        # Combine inputs and outputs (NO PADDING!)
        # Input: 8 (img) + 4 (time) + 2 (z) = 14 features
        input_vec = np.concatenate([
            img_normalized,
            time_normalized,
            z_normalized
        ])
        
        # Output: 4 (potent) + 8 (deflec) = 12 features
        output_vec = np.concatenate([
            potent_normalized,
            deflec_normalized
        ])
        
        inputs.append(input_vec)
        outputs.append(output_vec)
    
    return np.array(inputs), np.array(outputs)

# Keep legacy function for backward compatibility (with warning)
def normalize_test(dataset, max_images, scalers):
    """
    LEGACY FUNCTION: Use normalize_test_singles, normalize_test_doubles, or normalize_test_quads instead
    """
    print("Warning: Using legacy normalize_test function. Consider using specialized functions.")
    
    # Filter based on image count
    singles = [s for s in dataset if s['img'].shape[0] == 1]
    doubles = [s for s in dataset if s['img'].shape[0] == 2]
    quads = [s for s in dataset if s['img'].shape[0] == 4]
    
    all_inputs = []
    all_outputs = []
    
    # Process each type (this assumes the scalers are fitted on all types together)
    # This is not ideal but maintains backward compatibility
    for system in dataset:
        # Normalize each component using the EXISTING scalers
        img_normalized = scalers['img_scaler'].transform(system['img'])
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_2d = time_1d.reshape(-1, 1)
        time_normalized = scalers['time_scaler'].transform(time_2d).flatten()
        
        # Handle redshifts
        z_normalized = scalers['z_scaler'].transform([[system['zLens'], system['zSrc']]]).flatten()
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_2d = potent_1d.reshape(-1, 1)
        potent_normalized = scalers['potent_scaler'].transform(potent_2d).flatten()
        
        # Handle deflections
        deflec_normalized = scalers['deflec_scaler'].transform(system['deflec'])
        
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
        
        all_inputs.append(input_vec)
        all_outputs.append(output_vec)
    
    return np.array(all_inputs), np.array(all_outputs)