import math
import os
import constants as c
import numpy as np
import matplotlib.pyplot as plt
import pygravlens as gl
from astropy.cosmology import Planck18 as cosmo

def Generate_MockLens(args):
    
    shear_only, ellip_only, both = args

    # Validate input parameters
    valid_combinations = [
        (True, False, False),  # shear_only
        (False, True, False),  # ellip_only  
        (False, False, True)   # both
    ]
    
    if (shear_only, ellip_only, both) not in valid_combinations:
        raise ValueError("Error: Invalid parameter combination. Exactly one of \"shear_only\", \"ellip_only\", or \"both\" must be True.")

    values = []
    
    for i in range(c.num_mock):
        # Create lens plane based on the chosen model type
        if shear_only:
            plane_elpow = gl.lensplane('ellpow', [0.0, 0.0, 1.0, c.EinsArr[i], 0.0, 0.0], 
                                     gammac=c.gc[i], gammas=c.gs[i], Dl=c.Dlens[i])
        elif ellip_only:
            plane_elpow = gl.lensplane('ellpow', [0.0, 0.0, 1.0, c.EinsArr[i], c.ec[i], c.es[i]], 
                                     Dl=c.Dlens[i])
        elif both:
            plane_elpow = gl.lensplane('ellpow', [0.0, 0.0, 1.0, c.EinsArr[i], c.ec[i], c.es[i]], 
                                     gammac=c.gc[i], gammas=c.gs[i], Dl=c.Dlens[i])
        
        # Build the full lens model and tile it (if required)
        model_elpow = gl.lensmodel([plane_elpow], Ds=c.Dsrc[i])
        
        model_elpow.maingrid(-3,3,20,-3,3,20)
        model_elpow.galgrid(1.0e-6,3.0,20,20)
        
        model_elpow.tile(addlevels=0)
        
        # Compute the source position from the given image coordinates
        src_i, mag_tensor_i, time_del_i = model_elpow.lenseqn([c.im1[i], c.im2[i]]) # maps back one of the images to the source.
        
        # Find all images for that source
        imgarr, muarr, tarr = model_elpow.findimg(src_i) # gives back all the images that point to the source
        parr, defarr, garr = plane_elpow.defmag(imgarr)
        
        # defarr, marr = model_elpow.defmag(imgarr) deflection, magnification
        
        # Filter small magnification values
        boolean_mask = np.fabs(muarr) > (10**(-4))
        newImg_arr = imgarr[boolean_mask]
        newMag_arr = muarr[boolean_mask]
        newP_arr = parr[boolean_mask]
        newDef_arr = defarr[boolean_mask]
        newG_arr = garr[boolean_mask]
        newTime_arr = tarr[boolean_mask]
        
        # Extract source coordinates for dictionary
        beta_one, beta_two = src_i
        
        # Build dictionary with all relevant quantities
        if shear_only:
            elpow_dict = dict(
                img=newImg_arr, mu=newMag_arr, time=newTime_arr, potent=newP_arr, deflec=newDef_arr,
                ellipc=0.0, ellips=0.0, gammc=c.gc[i], gamms=c.gs[i],
                einrad=c.EinsArr[i], zLens=c.zlens[i], zSrc=c.zsrc[i],
                betaOne=beta_one, betaTwo=beta_two
            )
        elif ellip_only:
            elpow_dict = dict(
                img=newImg_arr, mu=newMag_arr, time=newTime_arr, potent=newP_arr, deflec=newDef_arr,
                ellipc=c.ec[i], ellips=c.es[i], gammc=0.0, gamms=0.0,
                einrad=c.EinsArr[i], zLens=c.zlens[i], zSrc=c.zsrc[i],
                betaOne=beta_one, betaTwo=beta_two  
            )
        elif both:
            elpow_dict = dict(
                img=newImg_arr, mu=newMag_arr, time=newTime_arr, potent=newP_arr, deflec=newDef_arr,
                ellipc=c.ec[i], ellips=c.es[i], gammc=c.gc[i], gamms=c.gs[i],
                einrad=c.EinsArr[i], zLens=c.zlens[i], zSrc=c.zsrc[i],
                betaOne=beta_one, betaTwo=beta_two
            )
        
        values.append(elpow_dict)
    
    return values
'''
### testing code locally

output_dir = "local_gen_test"
os.makedirs(output_dir, exist_ok=True)

try:
    vals_shear = Generate_MockLens([True, False, False])
    vals_ellip = Generate_MockLens([False, True, False])
    vals_both = Generate_MockLens([False, False, True])

    np.save(f'{output_dir}/valShear.npy', vals_shear)
    np.save(f'{output_dir}/valEllip.npy', vals_ellip)
    np.save(f'{output_dir}/valBoth.npy', vals_both)
    
    print(f"Mock lenses generated successfully {output_dir}/!")
    
except Exception as e:
    print(f"Error generating mock lenses: {e}")

'''
# Create output directory
output_dir = "AmarelOutput" 
os.makedirs(output_dir, exist_ok=True)

# Save code     
try:
    task_id = os.getenv('SLURM_ARRAY_TASK_ID', '0') # parallel processing aspect for Amarel
    
    vals_shear = Generate_MockLens([True, False, False])
    vals_ellip = Generate_MockLens([False, True, False])
    vals_both = Generate_MockLens([False, False, True])

    np.save(f'{output_dir}/valShear_{task_id}.npy', vals_shear)
    np.save(f'{output_dir}/valEllip_{task_id}.npy', vals_ellip)
    np.save(f'{output_dir}/valBoth_{task_id}.npy', vals_both)
    
    print(f"Mock lenses generated successfully {output_dir}/!")
    
except Exception as e:
    print(f"Error generating mock lenses: {e}")