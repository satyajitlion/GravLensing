### Today's Focus

Read up more from the OARC's cluster resources and work towards learning how to fix yesterday's errors involving installation, the SLURM script, and incorrect usage of Amarel directory.
***
### What I was able to accomplish

Took notes on several key operational aspects of the Amarel cluster:

- **Software Access:** Learned the correct commands (`module use`, `module load`, `module spider`) to find and load community-contributed software, which was a source of yesterday's installation errors.

- **Proper Job Execution:** Understood the critical practice of using `srun` for interactive work or a SLURM script for batch jobs to run on compute nodes, not login nodes. This directly addresses the root cause of yesterday's errors.

- **Job Management:** Gathered essential commands for monitoring (`squeue`), canceling (`scancel`), and analyzing job efficiency (`sacct`) after a job completes.

- **Partition Types:** Clarified the different SLURM partitions (main, gpu, mem, nonpre, etc.), which is crucial for writing an effective SLURM script that requests the right resources.

- **Installation Best Practices:** Learned the correct method to install Python packages locally using `python -m pip install [package] --user` after loading the desired `python` module, which should resolve the installation issues.

- **File Management:** Picked up a useful command (`rm -f *.out *.err *.pyc`) for cleaning up output files from failed jobs.
***
### Challenges & Pause Points

- Yesterday's problems are now clearly identified as a combination of:

	1. **Incorrect Node Usage:** Running installation/compilation directly on the login node.
    
	2. **Improper Installation Method:** Not using the correct `pip` command with the `--user` flag after loading a module.
    
	3. **Potential SLURM Script Issues:** The script may have been submitted to the wrong partition or did not request appropriate resources.
***
### Questions & Ideas

- **Question:** For my specific software (e.g., a Python machine learning library with GPU support), which `python` module version and which `gpu` partition should I use? Should I use `nonpre` to avoid preemption for long jobs?

- **Idea:** Create a template SLURM script that includes headers for the `gpu` partition, requests a specific GPU type, loads necessary modules, and uses the correct `pip` installation method within the script if needed.

- **Idea:** Test the installation process for my required packages by first getting an interactive session on a compute node with `srun --pty -p gpu -t 00:30:00 --gres=gpu:1 /bin/bash` and then running the installation commands there. This isolates any potential issues from the login node.
***
### Next Steps

1. **Test Installation Interactively:** Use `srun` to get an interactive session on a GPU compute node.

2. **Install Software Correctly:** In the interactive session, load the appropriate `python` module and install the necessary packages using `python -m pip install [package] --user`.

3. **Revise SLURM Script:** Update my SLURM batch script based on the partition information. Key changes will include:
    
	- Specifying the correct partition (e.g., `#SBATCH -p gpu`).

	- Requesting GPU resources (e.g., `#SBATCH --gres=gpu:1`).
    
	- Adding commands to load the required modules within the script.
    
	- Ensuring the job executes in the correct directory (e.g., `/scratch/ssg131/`).

4. **Submit a Test Job:** Run a short, small-scale test job with the revised script to verify everything works.

5. **Monitor and Analyze:** Use `squeue` to monitor the job and `sacct` after completion to check resource usage (like MaxRSS) for efficiency.

##### Tags: #Amarel #HPC #Terminal 




