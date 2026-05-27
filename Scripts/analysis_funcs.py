# imports all the modules needed
import math
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import arviz
from scipy import stats
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..', 'scripts'))
import pygravlens as gl
from astropy.cosmology import Planck18 as cosmo
from matplotlib.patches import Arc
from pathlib import Path 
from scipy.stats import skew
import scipy.stats as stats

#----------------------------------------#
# Helper functions (unchanged)
def ellipticity_magnitude(ec, es):
    return np.sqrt(ec**2 + es**2)

def shear_magnitude(gc, gs):
    return np.sqrt(gc**2 + gs**2)

#----------------------------------------#

def extract_parameters(vals, indices, has_shear=True):
    """Extract parameters from vals dict.
    Keys: 'ellipc', 'ellips', 'einrad', 'zLens', 'zSrc', 'gammc', 'gamms'
    """
    ellip = []
    einrad = []
    zLens = []
    zSrc = []
    shear = [] if has_shear else None

    for i in indices:
        d = vals[i]
        ellip.append(np.sqrt(d['ellipc']**2 + d['ellips']**2))
        einrad.append(d['einrad'])
        zLens.append(d['zLens'])
        zSrc.append(d['zSrc'])
        if has_shear:
            shear.append(np.sqrt(d['gammc']**2 + d['gamms']**2))

    out = {
        'ellip': np.array(ellip),
        'einrad': np.array(einrad),
        'zLens': np.array(zLens),
        'zSrc': np.array(zSrc)
    }
    if has_shear:
        out['shear'] = np.array(shear)
    return out

def get_indices_by_image_count(vals):
    single, double, quad = [], [], []
    for i, d in enumerate(vals):
        nimg = len(d['img'])
        if nimg == 1:
            single.append(i)
        elif nimg == 2:
            double.append(i)
        elif nimg == 4:
            quad.append(i)
    return double, quad

#-------------------------------------------------#

def plot_histograms(params_dict_list, labels, 
                    parameters=['ellip', 'einrad', 'zLens', 'zSrc', 'shear'],
                    bins=30, figsize=(15, 10)):
    n_params = len(parameters)
    ncols = (n_params + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, param in enumerate(parameters):
        ax = axes[idx]
        for p_dict, label in zip(params_dict_list, labels):
            if param in p_dict:
                data = p_dict[param]
                ax.hist(data, bins=bins, alpha=0.5, label=label, density=True)
        ax.set_xlabel(param)
        ax.set_ylabel('Density (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove unused subplots
    for i in range(len(parameters), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()

def inverse_transform_predictions(y_norm, scalers, n_images):
    """
    y_norm: (n_samples, n_features) normalized predictions
    scalers: dict with 'potent_scaler' and 'deflec_scaler'
    n_images: 2 for doubles, 4 for quads
    Returns:
        potent_phys: (n_samples, n_images) – potentials per image
        deflec_phys: (n_samples, n_images, 2) – deflections (x,y) per image
    """
    potent_scaler = scalers['potent_scaler']
    deflec_scaler = scalers['deflec_scaler']
    
    # --- Potentials ---
    # The scaler was fitted on each image's potential separately (shape: n_total_images, 1)
    potent_norm = y_norm[:, :n_images]                     # (n_samples, n_images)
    potent_flat = potent_norm.reshape(-1, 1)               # (n_samples * n_images, 1)
    potent_phys_flat = potent_scaler.inverse_transform(potent_flat)
    potent_phys = potent_phys_flat.reshape(-1, n_images)   # (n_samples, n_images)
    
    # --- Deflections ---
    # The scaler was fitted on individual deflection vectors (shape: n_total_images, 2)
    deflec_norm = y_norm[:, n_images:]                     # (n_samples, n_images*2)
    deflec_flat = deflec_norm.reshape(-1, 2)               # (n_samples * n_images, 2)
    deflec_phys_flat = deflec_scaler.inverse_transform(deflec_flat)
    deflec_phys = deflec_phys_flat.reshape(-1, n_images, 2)  # (n_samples, n_images, 2)
    
    return potent_phys, deflec_phys

#----------------------------------------------------------#

def compute_deflection_magnitude(deflec):
    """
    Compute magnitude of deflection vectors.
    
    Parameters
    ----------
    deflec : ndarray, shape (n_samples, n_images, 2)
        Deflection components (x,y) in physical units.
    
    Returns
    -------
    magnitude : ndarray, shape (n_samples, n_images)
        Magnitude sqrt(α_x² + α_y²) for each image.
    """
    return np.sqrt(deflec[..., 0]**2 + deflec[..., 1]**2)

def plot_output_histograms(pred_potent, true_potent, pred_deflec, true_deflec,
                           n_images, title_prefix="", bins=40, figsize=(15, 4)):
    """
    Plot overlaid histograms (predicted vs true) for potentials and deflections.
    
    Parameters
    ----------
    pred_potent : ndarray, shape (n_samples, n_images)
        Predicted potentials in physical units.
    true_potent : ndarray, shape (n_samples, n_images)
        True potentials in physical units.
    pred_deflec : ndarray, shape (n_samples, n_images, 2)
        Predicted deflections (x,y) in physical units.
    true_deflec : ndarray, shape (n_samples, n_images, 2)
        True deflections (x,y) in physical units.
    n_images : int
        Number of images (2 for doubles, 4 for quads).
    title_prefix : str
        Prefix for plot titles (e.g., "Doubles").
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure size.
    """
    # Flatten across samples and images
    pred_pot_flat = pred_potent.flatten()
    true_pot_flat = true_potent.flatten()
    pred_defx_flat = pred_deflec[..., 0].flatten()
    true_defx_flat = true_deflec[..., 0].flatten()
    pred_defy_flat = pred_deflec[..., 1].flatten()
    true_defy_flat = true_deflec[..., 1].flatten()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Helper to plot two histograms with same bins and density
    def hist_overlay(ax, data_pred, data_true, label_pred, label_true, xlabel, bins):
        # Use combined range to set consistent bins
        all_data = np.concatenate([data_pred, data_true])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)
        ax.hist(data_pred, bins=bin_edges, alpha=0.5, density=True, label=label_pred)
        ax.hist(data_true, bins=bin_edges, alpha=0.5, density=True, label=label_true)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Potential
    hist_overlay(axes[0], pred_pot_flat, true_pot_flat,
                 'Predicted', 'True', 'Potential', bins)
    axes[0].set_title(f'{title_prefix} Potentials')

    # Deflection x
    hist_overlay(axes[1], pred_defx_flat, true_defx_flat,
                 'Predicted', 'True', r'$\alpha_x$ (arcsec)', bins)
    axes[1].set_title(f'{title_prefix} Deflection x')

    # Deflection y
    hist_overlay(axes[2], pred_defy_flat, true_defy_flat,
                 'Predicted', 'True', r'$\alpha_y$ (arcsec)', bins)
    axes[2].set_title(f'{title_prefix} Deflection y')

    plt.tight_layout()
    plt.show()

def plot_output_histograms_magnitude(pred_potent, true_potent, 
                                    pred_deflec, true_deflec,
                                    n_images, title_prefix="", bins=40, 
                                    figsize=(10, 4)):

    # Plot overlaid histograms (predicted vs true) for potentials and deflection magnitude.
    # Parameters
    # ----------
    # pred_potent : ndarray, shape (n_samples, n_images)
    #     Predicted potentials in physical units.
    # true_potent : ndarray, shape (n_samples, n_images)
    #     True potentials in physical units.
    # pred_deflec : ndarray, shape (n_samples, n_images, 2)
    #     Predicted deflections (x,y) in physical units.
    # true_deflec : ndarray, shape (n_samples, n_images, 2)
    #     True deflections (x,y) in physical units.
    # n_images : int
    #     Number of images (2 for doubles, 4 for quads).
    # title_prefix : str
    #     Prefix for plot titles (e.g., "Doubles").
    # bins : int
    #     Number of histogram bins.
    # figsize : tuple
    #     Figure size (width, height).
    # Compute deflection magnitudes

    pred_mag = compute_deflection_magnitude(pred_deflec)
    true_mag = compute_deflection_magnitude(true_deflec)
        
    # Flatten across samples and images
    pred_pot_flat = pred_potent.flatten()
    true_pot_flat = true_potent.flatten()
    pred_mag_flat = pred_mag.flatten()
    true_mag_flat = true_mag.flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    def hist_overlay(ax, data_pred, data_true, label_pred, label_true, xlabel, bins):
        # Use combined range to set consistent bins
        all_data = np.concatenate([data_pred, data_true])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)
        ax.hist(data_pred, bins=bin_edges, alpha=0.5, density=True, label=label_pred)
        ax.hist(data_true, bins=bin_edges, alpha=0.5, density=True, label=label_true)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Potential
    hist_overlay(axes[0], pred_pot_flat, true_pot_flat,
                'Predicted', 'True', 'Potential', bins)
    axes[0].set_title(f'{title_prefix} Potentials')
    
    # Deflection magnitude
    hist_overlay(axes[1], pred_mag_flat, true_mag_flat,
                'Predicted', 'True', r'$|\vec{\alpha}|$ (arcsec)', bins)
    axes[1].set_title(f'{title_prefix} Deflection Magnitude')
    
    plt.tight_layout()
    plt.show()

# ------------------------------------------------ #

def plot_residual_histograms(pred_potent, true_potent, pred_deflec, true_deflec,
                             n_images, title_prefix="", bins=40, figsize=(10, 4)):
    """
    Plot histograms of residuals (predicted - true) for potentials and deflection magnitude.
    
    Parameters
    ----------
    pred_potent, true_potent : ndarray, shape (n_samples, n_images)
    pred_deflec, true_deflec : ndarray, shape (n_samples, n_images, 2)
    n_images : int
    title_prefix : str
    bins : int
    figsize : tuple
    """
    # Compute residuals (flattened)
    resid_pot = (pred_potent - true_potent).flatten()
    
    pred_mag = compute_deflection_magnitude(pred_deflec)
    true_mag = compute_deflection_magnitude(true_deflec)
    resid_mag = (pred_mag - true_mag).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Potential residuals
    axes[0].hist(resid_pot, bins=bins, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    axes[0].axvline(0, color='k', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel('Residual (Predicted - True) Potential')
    axes[0].set_ylabel('Density (Normalized)')
    axes[0].set_title(f'{title_prefix} Potential Residuals')
    axes[0].grid(True, alpha=0.3)
    
    # Magnitude residuals
    axes[1].hist(resid_mag, bins=bins, alpha=0.7, color='darkorange', edgecolor='black', density=True)
    axes[1].axvline(0, color='k', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel(r'Residual (Predicted - True) $|\vec{\alpha}|$ (arcsec)')
    axes[1].set_ylabel('Density (Normalized)')
    axes[1].set_title(f'{title_prefix} Deflection Magnitude Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== {title_prefix} Residual Statistics ===")
    print(f"Potential residuals:  mean = {np.mean(resid_pot):.4f}, std = {np.std(resid_pot):.4f}")
    print(f"Magnitude residuals: mean = {np.mean(resid_mag):.4f}, std = {np.std(resid_mag):.4f}")

def plot_per_image_histograms_magnitude(pred_potent, true_potent,
                                        pred_deflec, true_deflec,
                                        n_images, title_prefix="", bins=40):
    """
    Create a grid of histograms: rows = images, columns = (potential, |α|).
    Each histogram overlays predicted and true distributions for a single image index.
    """
    pred_mag = compute_deflection_magnitude(pred_deflec)
    true_mag = compute_deflection_magnitude(true_deflec)
    
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 4 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for img_idx in range(n_images):
        # Potential
        ax = axes[img_idx, 0]
        all_pot = np.concatenate([pred_potent[:, img_idx], true_potent[:, img_idx]])
        bin_edges = np.histogram_bin_edges(all_pot, bins=bins)
        ax.hist(pred_potent[:, img_idx], bins=bin_edges, alpha=0.5, density=True,
                label='Predicted', color='steelblue')
        ax.hist(true_potent[:, img_idx], bins=bin_edges, alpha=0.5, density=True,
                label='True', color='darkorange')
        ax.set_xlabel('Potential')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'{title_prefix} Image {img_idx+1} Potential')
        ax.grid(True, alpha=0.3)
        
        # Magnitude
        ax = axes[img_idx, 1]
        all_mag = np.concatenate([pred_mag[:, img_idx], true_mag[:, img_idx]])
        bin_edges = np.histogram_bin_edges(all_mag, bins=bins)
        ax.hist(pred_mag[:, img_idx], bins=bin_edges, alpha=0.5, density=True,
                label='Predicted', color='steelblue')
        ax.hist(true_mag[:, img_idx], bins=bin_edges, alpha=0.5, density=True,
                label='True', color='darkorange')
        ax.set_xlabel(r'$|\vec{\alpha}|$ (arcsec)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'Image {img_idx+1} Deflection Magnitude')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_qq_residuals(pred_potent, true_potent, pred_deflec, true_deflec,
                      n_images, title_prefix="", figsize=(10, 4), marker_style='o', markersize=1):
    """
    Create Q-Q plots for potential and deflection magnitude residuals.
    
    Parameters
    ----------
    marker_style : str, default ','
        Matplotlib marker style. ',' gives a pixel; '.' gives a small dot.
    markersize : float, default 1
        Size of the markers.
    """
    # Compute residuals
    resid_pot = (pred_potent - true_potent).flatten()
    
    pred_mag = np.sqrt(pred_deflec[..., 0]**2 + pred_deflec[..., 1]**2)
    true_mag = np.sqrt(true_deflec[..., 0]**2 + true_deflec[..., 1]**2)
    resid_mag = (pred_mag - true_mag).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Potential residuals Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(resid_pot, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set_marker(marker_style)
    axes[0].get_lines()[0].set_markersize(markersize)
    axes[0].get_lines()[0].set_linestyle('none')  # Remove connecting lines if any
    axes[0].get_lines()[0].set_color('steelblue')
    axes[0].get_lines()[1].set_color('black')     # Reference line
    axes[0].set_title(f'{title_prefix} Potential Residuals')
    axes[0].set_xlabel('Theoretical Quantiles')
    axes[0].set_ylabel('Ordered Residuals')
    
    # Magnitude residuals Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(resid_mag, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_marker(marker_style)
    axes[1].get_lines()[0].set_markersize(markersize)
    axes[1].get_lines()[0].set_linestyle('none')
    axes[1].get_lines()[0].set_color('darkorange')
    axes[1].get_lines()[1].set_color('black')
    axes[1].set_title(f'{title_prefix} Deflection Magnitude Residuals')
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Ordered Residuals')
    
    plt.tight_layout()
    plt.show()