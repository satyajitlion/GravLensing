# Application of Machine Learning in Gravitational Lens Modeling

A pipeline for generating mock gravitational lens data and training neural networks to constrain the Hubble Constant ($H_0$).

## Project Overview

This project develops a machine learning approach to gravitational lens modeling, with the goal of determining the Hubble Constant. The workflow consists of three main phases:

1. **Mock Lens Generation**: Create realistic training data using gravitational lensing physics
2. **Neural Network Training**: Train models to predict lens parameters from observable features
3. **Hubble Constant Estimation**: Apply trained models to real lens systems to constrain H₀

## Scientific Motivation

Gravitational lens time delays provide one of the most direct methods for measuring $H_0$, but traditional modeling approaches are computationally inefficient. This project explores machine learning as a faster, more scalable alternative for extracting cosmological parameters from lensed systems.

## Project Structure

| Folder / File | Description |
|---------------|-------------|
| **Notebooks/** | Jupyter notebooks for analysis, testing, and learning |
| &nbsp;&nbsp;&nbsp;`lensingBasics.ipynb` | Educational introduction to gravitational lensing concepts |
| &nbsp;&nbsp;&nbsp;`MockLensTests.ipynb` | Validates the dictionary structure and functionality of generated mock lenses |
| &nbsp;&nbsp;&nbsp;`analysis.ipynb` | Analyzes mock lens data (distributions, correlations, etc.) |
| &nbsp;&nbsp;&nbsp;`corner_plot_analysis.ipynb` | Creates corner plots to visualize parameter correlations |
| &nbsp;&nbsp;&nbsp;`timedelays.ipynb` | Explores time delay calculations and their relation to $H_0$ |
| &nbsp;&nbsp;&nbsp;`toy_model.ipynb` | Simple toy model for testing machine learning concepts before scaling up |
| **Scripts/** | Python scripts for data generation, combination, and modeling |
| &nbsp;&nbsp;&nbsp;`pygravlens.py` | Core gravitational lensing software (Dr. Keeton’s code) – provides lens plane and lens model classes, deflection calculations, image finding, etc. |
| &nbsp;&nbsp;&nbsp;`constants.py` | Configuration parameters and constants used across scripts: number of mock lenses, distributions for ellipticity/shear, Einstein radii, redshifts, and pre‑computed arrays for lens and source positions. |
| &nbsp;&nbsp;&nbsp;`generateMockLens.py` | Main generation script. Produces mock lenses (shear‑only, ellipticity‑only, and combined) and saves them as NumPy arrays. On Amarel, it uses `SLURM_ARRAY_TASK_ID` to parallelize output. |
| &nbsp;&nbsp;&nbsp;`combineMockLens.py` | Combines the parallelized output files from Amarel (e.g., `valShear_*.npy`) into single arrays (`valShear.npy`, etc.) saved in `combined_arrays/`. |
| &nbsp;&nbsp;&nbsp;`NetworkModel.py` | Neural network architecture and training routines. Contains separate model builders for single‑, double‑, and quad‑image lenses, plus custom callbacks (early stopping with minimum epochs) and augmentation functions (rotation/translation). |
| &nbsp;&nbsp;&nbsp;`extract_data.py` | Extracts and normalizes features from the mock lens dictionaries for neural network input. Separate functions for singles, doubles, and quads, each returning input/output arrays and fitted scalers. |
| &nbsp;&nbsp;&nbsp;`normalize_test_data.py` | Uses the scalers fitted during extraction to normalize new (test) data. Provides functions for singles, doubles, and quads to ensure consistent preprocessing. |
| **AmarelOutput/** | Output folder created by `generateMockLens.py` on the cluster. Contains parallelized arrays (e.g., `valShear_0.npy`, `valShear_1.npy`, …). |
| **combined_arrays/** | Output folder created by `combineMockLens.py` after merging the parallelized arrays. Contains the final datasets: `valShear.npy`, `valEllip.npy`, `valBoth.npy`. |
| **local_gen_test/** | Small local test folder used to verify `generateMockLens.py` produces the expected output. Contains test arrays generated during development. |
| **meeting_notes/** | Research meeting notes from Dr. Keeton (contains meeting summaries, discussions, and action items). |
| **Notes/** | Personal notes on mathematics and coding related to neural networks (e.g., derivations, architecture notes, debugging logs). |
| **research_log/** | (Old, to be removed) Research log documenting daily work. No longer actively maintained. |
| **requirements.txt** | List of required Python libraries; install with `pip install -r requirements.txt`. Includes scientific packages (NumPy, SciPy, astropy, matplotlib, etc.) and machine learning libraries (TensorFlow, Keras, scikit‑learn). |
| **Honors in Astronomy Project Outline.pdf** | Project proposal and documentation from September 2024 to present (originally for undergraduate honors, now continued as a post‑baccalaureate researcher). |
| **README.md** | This file. |

## Project Pipeline

### Phase 1: Data Generation (Completed)
- **Input**: Cosmological parameters, lens/source redshifts, mass distributions
- **Process**: Physical lens modeling using `pygravlens.py`
- **Output**: Mock lens systems with known parameters (our training data for ML model)

### Phase 2: Machine Learning Training (Current)
- **Input**: Mock lens data (image positions, magnifications, time delays)
- **Process**: Neural network training to learn parameter → observables mapping
- **Output**: Trained models that can predict parameters from observables

### Phase 3: Hubble Constant Application (Future)
- **Input**: Real lens systems (e.g., from H0LiCOW, TDCOSMO)
- **Process**: Apply trained ML models to infer $H_0$
- **Output**: Hubble Constant constraints with uncertainty estimates

## Requirements

### Core Dependencies
- Python 3.13.12
- Scientific Libraries (see `requirements.txt`)
- jupyter (for analysis notebooks)

### ML Dependencies
- tensorflow & keras
- scikit-learn
- pandas

### Lens Modeling Implementation
- `pygravlens.py` - Dr. Keeton's gravlens software

## Quick Start

### 1. Generate Training Data
```bash
# Generate mock lenses for ML training
python generateMockLenses.py
```

### 2. Analyze Data Properties
```
jupyter notebook analysis.ipynb
```

### 3. Validate Data Structure
```
jupyter notebook MockLensTests.ipynb
```

## Features (Network Input)
```
{
    'img': array,          # Image positions [RA, Dec] × N_images
    'mu': array,           # Magnifications for each image
    'time': array,         # Relative time delays
    # Derived features: image separation, magnification ratios, etc.
}
```

## Labels (Network Targets)
```
{
    'ellipc': float,       # Ellipticity component 1
    'ellips': float,       # Ellipticity component 2  
    'gammc': float,        # External shear component 1
    'gamms': float,        # External shear component 2
    'einrad': float,       # Einstein radius
    'zLens': float,        # Lens redshift
    'zSrc': float,         # Source redshift
    # H₀ is derived from these parameters + time delays
}
```

## Research Context
This work contributes to the Hubble tension. Machine learning approaches could help resolve systematic uncertainties in lens modeling.
