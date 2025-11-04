import numpy as np

def extract_lens_data(dataset, max_images):
    inputs = []
    outputs = []
    
    for system in dataset:
        n_images = len(system['img'])
        
        # Pad image positions: each image has (x,y) so we need max_images * 2 elements
        img_padded = np.zeros(max_images * 2)  # Create array of zeros for padding
        img_flat = system['img'].flatten()     # Flatten original image array: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
        img_padded[:len(img_flat)] = img_flat  # Fill beginning with actual data, rest stays zero
        
        # Pad time delays: each image has one time delay
        time_padded = np.zeros(max_images)                    # Create padding array
        time_flat = system['time'].to_value().flatten()       # Extract numeric values and flatten
        time_padded[:len(time_flat)] = time_flat              # Fill with actual time data
        
        # Pad potentials: each image has one potential value
        potent_padded = np.zeros(max_images)                  # Create padding array  
        potent_flat = system['potent'].flatten()              # Flatten potentials
        potent_padded[:len(potent_flat)] = potent_flat        # Fill with actual potential data
        
        # Pad deflections: each image has (deflection_x, deflection_y)
        deflec_padded = np.zeros(max_images * 2)              # Create padding array
        deflec_flat = system['deflec'].flatten()              # Flatten deflections
        deflec_padded[:len(deflec_flat)] = deflec_flat        # Fill with actual deflection data
        
        # Combine all input components into one feature vector
        # Format: [padded_images, padded_times, z_lens, z_src]
        input_vec = np.concatenate([img_padded, time_padded, [system['zLens'], system['zSrc']]])
        
        # Combine all output components into one target vector  
        # Format: [padded_potentials, padded_deflections]
        output_vec = np.concatenate([potent_padded, deflec_padded])
        
        inputs.append(input_vec)
        outputs.append(output_vec)
    
    return np.array(inputs), np.array(outputs)
