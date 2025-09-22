# Gravitational Lens Mock Data Generator

A Python tool for generating mock gravitational lensing data using pygravlens.

## Project Overview

This code generates three types of mock gravitational lens systems:
- **Shear-only lenses**: External shear only, no ellipticity
- **Ellipticity-only lenses**: Intrinsic ellipticity only, no external shear  
- **Combined lenses**: Both ellipticity and external shear

## Requirements

### Python Packages
- Python 3.7+
- numpy
- matplotlib
- pygravlens
- astropy
- scipy (optional)

### System Requirements
- Minimum: 8GB RAM for small datasets (≤1,000 lenses)
- Recommended: 16GB+ RAM for large datasets (10,000+ lenses)
- Storage: ~100MB per 10,000 lenses generated

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
cd gravitational-lens-generator
```

2. Clone this repository:
```pip install numpy matplotlib astropy
# Install pygravlens according to its documentation
```

# File Structure
project/
├── generateMockLens.py     # Main generation script
├── constants.py           # Configuration parameters
├── README.md             # This file
├── valShear.npy          # Output: shear-only lenses
├── valEllip.npy          # Output: ellipticity-only lenses
└── valBoth.npy           # Output: combined lenses

3. 