### Accessing Libraries and Applications

```terminal
module use /projects/community/modulefiles
module load <name/version>
module spider
module avail
```

1. Community contributed software: https://sites.google.com/view/cluster-user-guide/amarel/community?authuser=0
2. Use srun to connect to a "compute node" instead of working through a login-node. Not only is it bad  practice, but it leads to errors like the ones I made yesterday.

```terminal
sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID
```

##### Check / track batch job

```terminal
squeue -u 'ssg131'
```

##### Cancel batch job

```terminal
scancel jobID
```

##### Output contents from a file for quick view

```terminal
cat FILE_NAME
```

##### Partition types (for type of batch job):
- main - traditional compute nodes, CPUs only, jobs running here are preemptible
- gpu - nodes with any of our large collection of general-purpose GPU accelerators
- mem - CPU-only nodes with 512 GB to 1.5 TB RAM
- nonpre - a partition where jobs won't be preempted by higher-priority or owner jobs
- graphical - a specialized partition for jobs submitted by the OnDemand system 
- cmain - the "main" partition for the Amarel resources located in Camden, note that /scratch for those nodes is within the Amarel-C system

##### Quick note:
```terminal
rm -f *.out *.err *.pyc
```
The above removes all of the files with a .out, .err, or .pyc ending and this is pretty useful for getting rid of failed job outputs.

```terminal
wget [https://www.link-to-software.tar.gz]
tar [installed file with a .tar.gz ending]
```

##### To install things:

```terminal
module load python
python -m pip install [package] --user
```

The above could be really helpful for installing the required software libraries for my code.