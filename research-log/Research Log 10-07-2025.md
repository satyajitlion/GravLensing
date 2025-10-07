### Today's Focus

Transition from theoretical neural network research back to the generation of Mock Gravitational Lenses using Amarel. Find the root cause of errors with Dr. Keeton and finish setting up the computational pipeline on Amarel cluster to generate mock lensing data for future neural network training, resolving environment configuration issues and establishing reliable batch processing workflows. Ensure that the local environment setup on Amarel is consistent in terms of the versions of it's installed modules and libraries. Finally, submit the batch job after fixing all the configuration related issues. Ensure that the batch job doesn't fail due to any other errors and double check the SLURM script to ensure that the batch job will run smoothly.
***
### What I was able to accomplish

- **Resolved critical NumPy compatibility issue** that was preventing pygravlens from running by creating a dedicated conda environment with compatible package versions (numpy=1.23.5, scipy=1.9.3)
    
- **Successfully configured the complete gravitational lensing simulation environment** with proper dependencies including shapely, astropy, matplotlib, scipy, and etc.
    
- **Implemented and tested the mock lens generation pipeline** with three mass model configurations in `generateMockLens.py`:
    
    - Shear-only models
        
    - Ellipticity-only models
        
    - Combined shear+ellipticity models

- **Successfully submitted the batch job** and am currently waiting for the batch job to finish over the next 24 hours.
***
### Results

- Created a stable, reproducible computational environment that resolves the dependency issues encountered over the course of these few weeks in Amarel (mainly issues with version control).

- Successfully submitted `mock_lens_100k` batch job generating three datasets of 100,000 gravitational lenses each

- Developed reliable process for job submission, monitoring, and management on HPC cluster
***
### Challenges & Pause Points

- **Initial Package Conflicts**: Spent significant time resolving NumPy/SciPy version incompatibilities that prevented pygravlens from importing
    
- **SLURM Learning Curve**: Required trial and error to properly configure batch scripts and understand job management commands
    
- **Editor Artifacts**: Encountered and resolved Emacs auto-save files (`#pygravlens.py#`) in working directory
    
- **Pending Completion**: The 100,000 lens simulation job is currently running - results and data quality verification are pending
***
### Questions & Ideas

- Should we implement data validation checks to ensure the generated mock lenses are physically realistic before using them for training?

- How do I ensure that for the Neural Network, translations and rotations of the system don't change the pattern that the network develops between the inputs and the outputs? 
	- <u>Idea</u>: Maybe use Tensors? Tensors have changing elements but a static metric that characterizes the system. But how do I do this? Will need to learn more here.
	
***
### Next Steps

##### Tags:




