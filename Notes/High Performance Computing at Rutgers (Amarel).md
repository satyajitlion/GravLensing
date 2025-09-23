### Notes:

- ##### OARC - office of advanced research computing
	- University wide research computing support team
	- manages multiple clusters and storage systems
	- computational scientists are available for consultation and trainig
	- They can help with developing proposals that make use of  Rutgers research computing resources: computing, storage,  networking, cloud services, etc.
		- email: help@oarc.rutgers.edu
	
- ##### Research Computing
	- Need for parallel computing or management of "big data" and/or computation exceeds capabilities of local workstations which is where HPC/HTC (High-PerformanceComputing and High-Throughput Computing) come into play
		- Local work stations: 4-16 GB RAM
		- Advanced computing systems: 16 GB - 2 TB RAM
	
- ##### Using a Computer Cluster
	1. Connect to a cluster (via SSH), setup your software to run there
	2. Move and input files/data to the cluster (via rsync, scp, sftp)
	3. Create a job script (that requests only the hardware one needs)
	4. Submit the job script to cluster's resource manager 
	5. After job has finished running, collect the output files.
	
- ##### Logging into Amarel
	- In windows, launch a SSH client (PuTTY or MobaXterm)
		- hostname: amarel.rutgers.edu
		- Click "Open"
		- Login as: [insert my NETID]
	- **Important NOTE:** if you are off campus, then you need to connect to a vpn first.
		- https://soc.rutgers.edu/vpn
	
- ##### The /home Filesystem
	- Located at /home/NetID
	- 100 GB storage space
	- Backed up storage
	- space to install customized software 
	
- ##### The /scratch Filesystem
	- Located at /scratch/NetID
	- Temporary work directory for all jobs
	- Specialized high-performance hardware
	- Designed to handle high I/O activity and large files
	- moves files in, runs jobs, moves files out
	- 1 TB storage space (2 TB hard limit with 2 week-grace period)
	- Non-backed up and subjected to purge (files older than 90 days).
	
- ##### Resource Management
	- SLURM = resource manager / job scheduler
	- Enables scripting of computational tasks
	- Slurm runs these tasks on compute nodes and returns the results (output files)
	- If the cluster is full, SLURM holds tasks and runs them when the resources are available
	- SLURM ensures fair sharing of cluster resources (policy enforcement)
	
- ##### Basic SLURM commands
	- All commands: https://slurm.schedmd.com/pdfs/summary.pdf

| # Commands                        | # Description                                        |
|-----------------------------------|------------------------------------------------------|
| sinfo -a                          | Views Nodes and partition info                       |
| sbatch job-script [options]       | Submit/setup a batch job                             |
| srun [options] program_name       | Run a program (exe, application)                     |
| squeue -u NetID                   | Check status of job submissions                      |
| sstat -u jobID                    | Check status of a running job                        |
| sacct --format [options] -j jobID | See accounting details of current and completed jobs |

- ##### Difference between ***batch job*** and ***interactive job*** 	
| # Batch Job                                                                                                                                                 | # Interactive Job                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **sbatch** *job-script*                                                                                                                                     | **srun** [required resources] *your.exe*                                         |
| Starts when requested resources are available                                                                                                               | Starts when requested resources are available                                    |
| Runs "in the background" (Use **srun** to launch tasks inside your script).                                                                                 | You are actively logged-in (running a shell) on a compute node.                  |
| Terminate batch jobs using **scancel** jobID                                                                                                                | Terminate interactive jobs by simply logging-out (using **exit** or **CNTRL+D**) |
| Useful for jobs that will run for a long time (essentially my case for the research I'm doing), and for jobs that don't require interaction or supervision. | Useful for testing, compiling code, computational steering, etc.                 |
- ##### Batch Job Script
```
#!/bin/bash
#SBATCH --partition=main  
#SBATCH --job-name=Efexa  
#SBATCH --nodes=1  
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=16  
#SBATCH --exclusive  
#SBATCH --mem=118G  
#SBATCH --time=5:00:00  
#SBATCH --output=slurm.%N.%j.out  
#SBATCH --error=slurm.%N.%j.err  
#SBATCH --mail-user=[NetID]@rutgers.edu  
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --export=ALL

srun my-exe –t 16 input > output  
sacct --format MaxRSS,Elapsed -j $SLURM_JOBID  
```
- note that here, "--mem=0" means "use all available RAM" but here, that's only the RAM that's not allocated to other jobs.

- ##### Copying Example Files
```
cd /scratch/[NetID]
cp -r /projects/oarc/users/training/intro.amarel .  
cd intro.amarel  
ls
```

- ##### Example Job Script
```
#!/bin/bash  
#SBATCH --clusters=amarel # Select which system(s) to use  
#SBATCH --partition=main # Partition (job queue)  
#SBATCH --requeue # Return job to the queue if preempted  
#SBATCH --job-name=LAMMPSX # Assign a short name to your job  
#SBATCH --nodes=1 # Number of nodes you require  
#SBATCH --ntasks=8 # Total # of tasks across all nodes  
#SBATCH --cpus-per-task=1 # Cores per task (>1 if multithread tasks)  
#SBATCH --mem=1G # Real memory (RAM) required per node  
#SBATCH --time=10:00:00 # Total run time limit (DD-HH:MM:SS)  
#SBATCH --output=slurm.%N.%j.out # STDOUT file for SLURM output  

## Environment settings needed for this job  
module purge  
module load intel/17.0.4 mvapich2/2.2

## Run the job  
srun --mpi=pmi2 lammps_intel_cpu_mpi < in.binary  

## Capture job accounting info (OPTIONAL)  
sleep 10  
sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID | sed -n -e 1,2p -e 5p
```

- ##### Defaults and Limits
	```
	--ntasks=1
	--cpus-per-task=1
	--mem-per-cpu=4G
	--time=00-00:2:00
	```

	- Limits / Upper Bounds:
		- Max # CPUs in-use per user for 'main' partition jobs = 504
		- Max # of jobs a user can submit = 500 (this also applies to job arrays), owners have a 2000 job limit for each partition
	
	- Consider memory request boundaries (ask for about 8 GB less than the max)

- ##### Amarel Cluster Website
	- https://sites.google.com/view/cluster-user-guide



##### tag: #Amarel #High-PerformanceComputing #Terminal 