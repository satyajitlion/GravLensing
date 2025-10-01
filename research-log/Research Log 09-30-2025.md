### Today's Focus

Resolve the critical Python dependency issues on the Amarel cluster, specifically for the installation of the NumPy, SciPy, Astropy, Matplotlib, and shapely packages, to enable the successful execution of the mock lens generation SLURM batch job.
***
### What I was able to accomplish
- Discovered and loaded the `py-data-science-stack` module, which provided a pre-configured environment with Python and most required packages (NumPy, SciPy, Astropy, Matplotlib).
- Found the root cause: the system's Python 3.8.2 installation is fundamentally broken (missing `_ctypes` module), making scientific computing impossible without using specialized modules.
- Successfully resolved library conflicts by cleaning up mixed installations and ensuring all packages work within the `py-data-science-stack` environment.
- Finally achieved a stable, working Python environment where all dependencies (including the problematic `shapely` package) can be properly imported.
***
### Results

- The core issue was a broken system Python installation, not just missing packages. This explains why initial installation attempts failed.
- The `py-data-science-stack` module provides a properly configured environment for scientific computing on Amarel (with everything besides shapely).
- After cleaning up installation conflicts, most required packages were able to be imported (besides SciPy for Python 3.8.2 and Shapely for the built in python version given by the `py-data-science-stack` module).
***
### Problems

- The system's Python 3.8.2 installation is missing critical components (`_ctypes` module), making it impossible to use scientific packages like SciPy. This explains why initial dependency installations failed. Additionally, my directory was using python version 2.7.5 or defaulting to it whenever I used to the `python [file_name]` command. 
- When switching to the working `py-data-science-stack` module, there were conflicts with previously installed packages that had broken library links.
- Required significant troubleshooting to clean up mixed installations and ensure consistency between interactive sessions and batch job environments.
- Shapely is still failing for the `py-data-science-stack` module which is why I am still unable to run my batch job without errors.
***
### Next Steps

- Reach out the Amarel support and explain the persisting error or leave this on pause for a bit?
- Immediately resubmit the mock lens generation batch job to begin producing the 100,000 lenses after installation related errors get fixed.
- Use `squeue` and `sacct` to track the job's status and resource usage.
- Start investigating ANN, RNN, and CNN architectures for the next phase while the batch job runs.
- Record the successful environment setup process for future reference and reproducibility.
##### Tags: #Amarel #HPC #Python #Dependencies #Troubleshooting #SLURM




