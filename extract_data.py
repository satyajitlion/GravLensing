import numpy as np

def extract_lens_data(dataset):
    """
    Extract inputs and outputs from lens dataset
    Returns: x_data (observable inputs), y_data (expected outputs)
    """
    
    inputs = []
    outputs = []
    
    for system in dataset:
        # Inputs: observables
        img_flat = system['img'].flatten()
        
        # Handle Astropy Quantity for time (extract values and convert to float)
        time_data = system['time']
        if hasattr(time_data, 'value'):
            time_flat = time_data.value.flatten()  # Extract numeric values
        else:
            time_flat = time_data.flatten()
        
        input_vec = np.concatenate([
            img_flat,                    # image positions
            time_flat,                   # time delays (now unitless)
            [system['zLens']],           # lens redshift
            [system['zSrc']]             # source redshift
        ])
        
        # Outputs: lens parameters  
        output_vec = np.concatenate([
            system['potent'].flatten(),  # potentials
            system['deflec'].flatten()   # deflection angles
        ])
        
        inputs.append(input_vec)
        outputs.append(output_vec)
    
    return np.array(inputs), np.array(outputs)