### Today's Focus

Resolve the critical Python dependency issues on the Amarel cluster, specifically for the installation of the NumPy, SciPy, Astropy, Matplotlib, and shapely packages, to enable the successful execution of the mock lens generation SLURM batch job.
***
### What I was able to accomplish

- Discovered and loaded the `py-data-sci` module, which I found on OARC's website. This provided a pre-configured environment with Python that was preloaded and contained most required packages (NumPy, SciPy, Astropy, Matplotlib), bypassing the need for individual installations.
- Confirmed that `shapely` was the only missing dependency not included in the `py-data-sci` bundle. Standard `pip install` commands failed due to missing system-level GEOS libraries.
- **Successfully Compiled and Installed Shapely from Source:** In a significant troubleshooting achievement, I:
    1. Located and loaded the necessary `gcc` and `cmake` modules to access a C/C++ compiler.
    2. Downloaded, compiled, and installed the GEOS library from source into a local `libraries` directory in my project folder.
    3. Configured the `shapely` build to use this local GEOS installation by setting the `GEOS_CONFIG` environment variable.
    4. Successfully built and installed `shapely` from its source distribution using `pip`.
- **Environment Validation:** Created and ran a test script (`test_imports.py`) that successfully imported all dependencies, including `shapely`, confirming the environment was fully functional
***
### Results

***
### Problems
- The system's Python 3.8.2 installation is missing critical components (`_ctypes` module), making it impossible to use scientific packages like SciPy. This explains why initial dependency installations failed.
- When switching to the working `py-data-sci` module, there were conflicts with previously installed packages that had broken library links.
***
### Questions & Ideas

***
### Next Steps

##### Tags:




