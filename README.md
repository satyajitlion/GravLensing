# Application of Machine Learning in Gravitational Lens Modeling

A pipeline for generating mock gravitational lens data and training neural networks to constrain the Hubble Constant ($H_0$).

## Project Overview

This project develops a machine learning approach to gravitational lens modeling, with the ultimate goal of determining the Hubble Constant. The workflow consists of three main phases:

1. **Mock Lens Generation**: Create realistic training data using gravitational lensing physics
2. **Neural Network Training**: Train models to predict lens parameters from observable features
3. **Hubble Constant Estimation**: Apply trained models to real lens systems to constrain H₀

## Scientific Motivation

Gravitational lens time delays provide one of the most direct methods for measuring $H_0$, but traditional modeling approaches are computationally inefficient. This project explores machine learning as a faster, more scalable alternative for extracting cosmological parameters from lensed systems.

## File Structure
| File/Folder | Description |
|-------------|-------------|
| `pygravlens.py` | Dr. Keeton's gravlens software |
| `generateMockLenses.py` | Main generation script |
| `constants.py` | Configuration parameters and constants |
| `analysis.ipynb` | Jupyter notebook for analyzing mock lenses |
| `MockLensTests.ipynb` | Testing dictionary structure and functionality |
| `lensingBasics.ipynb` | Educational notebook for lensing basics |
| `timedelays.ipynb` | Educational notebook for learning about time delays |
| `Honors in Astronomy Project Outline.pdf` | Project proposal/documentation |
| `valShear.npy` | Output: shear-only lenses |
| `valEllip.npy` | Output: ellipticity-only lenses |
| `valBoth.npy` | Output: combined lenses |
| `Meeting-Notes/` | Folder of notes from research meetings |
| `Research Log/` | Folder documenting my work |
| `Notes/` | Folder of notes I took for this research |
| `README.md` | Project documentation |

## Project Pipeline

### Phase 1: Data Generation (Completed)
- **Input**: Cosmological parameters, lens/source redshifts, mass distributions
- **Process**: Physical lens modeling using `pygravlens.py`
- **Output**: Mock lens systems with known parameters (→ training data)

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
- Python 3.7+
- numpy, matplotlib, astropy
- jupyter (for analysis notebooks)

### ML Dependencies (Future Phase)
- tensorflow/pytorch
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
