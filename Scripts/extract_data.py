import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_lens_data(dataset, max_images):
    inputs = []
    outputs = []
    
    # Initialize scalers
    img_scaler = StandardScaler()
    time_scaler = StandardScaler()
    z_scaler = StandardScaler()
    potent_scaler = StandardScaler()
    deflec_scaler = StandardScaler()
    
    # First pass: collect all data for fitting scalers (FAST VERSION)
    all_images = []
    all_times = []
    all_redshifts = []
    all_potentials = []
    all_deflections = []
    
    for system in dataset:
        all_images.append(system['img'])
        
        # FAST: Keep as arrays, handle reshaping later
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        all_times.append(time_1d)  # Keep as array
        
        all_redshifts.append([system['zLens'], system['zSrc']])
        
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        all_potentials.append(potent_1d)  # Keep as array
        
        all_deflections.append(system['deflec'])
    
    # FAST SCALER FITTING: Use concatenate instead of list operations
    img_scaler.fit(np.vstack(all_images))
    time_scaler.fit(np.concatenate(all_times).reshape(-1, 1))  # Fast concatenate
    z_scaler.fit(np.array(all_redshifts))
    potent_scaler.fit(np.concatenate(all_potentials).reshape(-1, 1))  # Fast concatenate
    deflec_scaler.fit(np.vstack(all_deflections))
    
    # Second pass: normalize and process (same as before)
    for system in dataset:
        # Normalize each component
        img_normalized = img_scaler.transform(system['img'])
        
        # Handle time data
        time_data = system['time'].to_value()
        time_1d = np.atleast_1d(time_data)
        time_2d = time_1d.reshape(-1, 1)
        time_normalized = time_scaler.transform(time_2d).flatten()
        
        z_normalized = z_scaler.transform([[system['zLens'], system['zSrc']]]).flatten()
        
        # Handle potentials
        potent_data = system['potent']
        potent_1d = np.atleast_1d(potent_data)
        potent_2d = potent_1d.reshape(-1, 1)
        potent_normalized = potent_scaler.transform(potent_2d).flatten()
        
        deflec_normalized = deflec_scaler.transform(system['deflec'])
        
        # Pad normalized data
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
    
    return np.array(inputs), np.array(outputs), {
        'img_scaler': img_scaler,
        'time_scaler': time_scaler, 
        'z_scaler': z_scaler,
        'potent_scaler': potent_scaler,
        'deflec_scaler': deflec_scaler
    }