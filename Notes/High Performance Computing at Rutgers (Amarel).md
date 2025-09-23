### Notes:

- OARC - office of advanced research computing
	- University wide research computing support team
	- manages multiple clusters and storage systems
	- computational scientists are available for consultation and trainig
	- They can help with developing proposals that make use of  Rutgers research computing resources: computing, storage,  networking, cloud services, etc.
		- email: help@oarc.rutgers.edu
	
- Research Computing
	- Need for parallel computing or management of "big data" and/or computation exceeds capabilities of local workstations which is where HPC/HTC (High-PerformanceComputing and High-Throughput Computing) come into play
		- Local work stations: 4-16 GB RAM
		- Advanced computing systems: 16 GB - 2 TB RAM
	
- Using a Computer Cluster:
	1. Connect to a cluster (via SSH), setup your software to run there
	2. Move and input files/data to the cluster (via rsync, scp, sftp)
	3. Create a job script (that requests only the hardware one needs)
	4. Submit the job script to cluster's resource manager 
	5. After job has finished running, collect the output files.
	
- Logging into Amarel:
	- In windows, launch a SSH client (PuTTY or MobaXterm)
		- hostname: amarel.rutgers.edu
		- Click "Open"
		- Login as: [insert my NETID]


##### tag: #Amarel, #High-PerformanceComputing, #Terminal 