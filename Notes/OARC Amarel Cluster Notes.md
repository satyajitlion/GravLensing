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