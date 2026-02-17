import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_single_lens_data(dataset):
    """Extract data for single image lenses (1 image)"""
    inputs = []
    outputs = []
    
    # Filter for systems with exactly 1 image
    single_systems = [s for s in dataset if s['img'].shape[0] == 1]
    
    # Initialize scalers
    img_scaler = StandardScaler()
    time_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()
    
    # First pass: collect all data for fitting scalers
    all_images = []
    all_times = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []
    
    for system in single_systems:
        all_images.append(system['img'])
        
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        all_times.append(time_1d)
        
        all_redshifts.append([system['zLens'], system['zSrc']])
        
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)
        
        all_deflections.append(system['deflec'])
    
    # Fit scalers
    if all_images:
        img_scaler.fit(np.vstack(all_images))
        time_scaler.fit(np.concatenate(all_times).reshape(-1, 1))
        z_scaler.fit(np.array(all_redshifts))
        potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))
        deflec_scaler.fit(np.vstack(all_deflections))
    
    # Second pass: normalize and process
    for system in single_systems:
        # Normalize each component
        img_normalized = img_scaler.transform(system['img']).flatten()  # Shape: (2,)
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (1,)
        
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (1,)
        
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
    
    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'time_scaler': time_scaler, 
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }

def extract_double_lens_data(dataset):
    """Extract data for double image lenses (2 images)"""
    inputs = []
    outputs = []
    
    # Filter for systems with exactly 2 images
    double_systems = [s for s in dataset if s['img'].shape[0] == 2]
    
    # Initialize scalers
    img_scaler = StandardScaler()
    time_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()
    
    # First pass: collect all data for fitting scalers
    all_images = []
    all_times = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []
    
    for system in double_systems:
        all_images.append(system['img'])
        
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        all_times.append(time_1d)
        
        all_redshifts.append([system['zLens'], system['zSrc']])
        
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)
        
        all_deflections.append(system['deflec'])
    
    # Fit scalers
    if all_images:
        img_scaler.fit(np.vstack(all_images))
        time_scaler.fit(np.concatenate(all_times).reshape(-1, 1))
        z_scaler.fit(np.array(all_redshifts))
        potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))
        deflec_scaler.fit(np.vstack(all_deflections))
    
    # Second pass: normalize and process
    for system in double_systems:
        # Normalize each component
        img_normalized = img_scaler.transform(system['img']).flatten()  # Shape: (4,)
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (2,)
        
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (2,)
        
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
    
    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'time_scaler': time_scaler, 
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }

def extract_quad_lens_data(dataset):
    """Extract data for quad image lenses (4 images)"""
    inputs = []
    outputs = []
    
    # Filter for systems with exactly 4 images
    quad_systems = [s for s in dataset if s['img'].shape[0] == 4]
    
    # Initialize scalers
    img_scaler = StandardScaler()
    time_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()
    
    # First pass: collect all data for fitting scalers
    all_images = []
    all_times = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []
    
    for system in quad_systems:
        all_images.append(system['img'])
        
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        all_times.append(time_1d)
        
        all_redshifts.append([system['zLens'], system['zSrc']])
        
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)
        
        all_deflections.append(system['deflec'])
    
    # Fit scalers
    if all_images:
        img_scaler.fit(np.vstack(all_images))
        time_scaler.fit(np.concatenate(all_times).reshape(-1, 1))
        z_scaler.fit(np.array(all_redshifts))
        potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))
        deflec_scaler.fit(np.vstack(all_deflections))
    
    # Second pass: normalize and process
    for system in quad_systems:
        # Normalize each component
        img_normalized = img_scaler.transform(system['img']).flatten()  # Shape: (8,)
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_normalized = time_scaler.transform(time_1d.reshape(-1, 1)).flatten()  # Shape: (4,)
        
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()  # Shape: (2,)
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_normalized = potent_scaler.transform(potent_1d.reshape(-1, 1)).flatten()  # Shape: (4,)
        
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
    
    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'time_scaler': time_scaler, 
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }