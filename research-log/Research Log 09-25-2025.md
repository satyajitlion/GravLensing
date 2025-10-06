### Today's Focus

Gain physical access to the research workstation (Room 330, Serin Physics Building) and establish a connection to the Amarel cluster. The primary objectives were to transfer project files to the cluster, create a dedicated project directory, and successfully submit a batch job via the SLURM workload manager to begin generating mock gravitational lenses.
***
### What I was able to accomplish

- **Access & Setup:** Successfully accessed the Amarel cluster via SSH directly from Git Bash, eliminating the need for PuTTy. Created a well-organized project structure by making a `GravLensing` subdirectory within my home directory.

- **File Management:** Used WinSCP to securely transfer all necessary Python scripts (`pygravlens.py`, `constants.py`, `generateMockLens.py`) from my local machine to the `GravLensing` directory on Amarel. Updated the `num_mock` parameter to $10^5$ in preparation for a large-scale run.

- **SLURM Job Submission:** After extensive consultation of OARC documentation and external resources to overcome a lack of course materials, I composed a functional SLURM script (`run_lens_job.slurm`). The script was configured with appropriate resource requests (24-hour walltime, 8 GB RAM) for the Amarel cluster. I submitted the job successfully using the `sbatch` command.

This process involved significant troubleshooting and independent learning to understand SLURM script syntax and Amarel's specific configuration, representing a major step forward in using HPC resources.
***
### Results

- Established a reliable workflow for accessing Amarel and managing project files.

- Created and submitted a SLURM batch job intended to generate 100,000 mock lenses.

- Identified a critical roadblock: the job failed immediately due to missing Python dependencies, halting progress on the primary objective.
***
### Problems

The main obstacle encountered was a **dependency management issue**. The batch job failed because the required Python libraries (NumPy, Astropy, SciPy, Shapely, Matplotlib) were not installed in my Amarel environment.

A secondary issue was **Python version configuration**. The system default was Python 2.7.5, while my code requires Python 3.8.2. I had to manually locate and load the correct version using the `module` system, indicating my environment needs proper configuration for consistency.
***
### Questions & Ideas

- **Question:** What is the best practice for installing Python packages on Amarel? Should they be installed in my home directory, or is there a project-specific or shared location?

- **Idea:** While the lens generation job runs (which will take time), I can pivot to the next phase of the project: researching and prototyping the neural network component using Keras/TensorFlow.

- **Idea:** I need to conduct a comparative analysis of ANN, RNN, and CNN architectures to determine the most suitable model for analyzing the generated lensing images.
***
### Next Steps

1. **Resolve Dependencies:** Priority #1 is to correctly install the missing Python packages. I will consult OARC documentation on Python package management and ensure installations are done on a compute node, not the login node.

2. **Seek Guidance:** If I cannot resolve the installation issue quickly, I will consult with Dr. Burkhart or submit a ticket to the OARC help desk.

3. **Parallel Work:** Once the batch job is queued, I will begin studying neural network implementations with Keras to maintain project momentum.

4. **Architecture Research:** Initiate a literature review to decide on the optimal neural network architecture (ANN, RNN, CNN) for gravitational lens analysis.

##### Tags: #Amarel #HPC #MockLens #MockLensGeneration #Terminal 




