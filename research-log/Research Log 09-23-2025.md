### Today's Focus

Establish the foundational setup required to use the Amarel cluster. The primary objectives were to understand the cluster's operation, install necessary software (PuTTy, WinSCP, Emacs), and establish a connection method. The ultimate goal was to log in, create a project directory for the gravitational lensing research, and transfer the initial set of files.
***
### What I was able to accomplish

- **Knowledge Acquisition:** Spent significant time (5-6 hours) reviewing Computational Astrophysics course materials and OARC documentation to compile essential notes on SLURM workload manager commands and High-Performance Computing (HPC) principles specific to Amarel.    
- **Software Setup:** Successfully downloaded and installed the core software toolkit for cluster interaction:
    - **Emacs:** For efficient text editing directly on the cluster.
    - **WinSCP:** For secure graphical file transfer between my local machine and Amarel.
    - **PuTTy:** As a backup SSH client for terminal access.
- **Research & Evaluation:** Researched alternative tools like MobaXterm and WSL to ensure a robust setup, deciding on the aforementioned software stack for initial attempts.

This day was dedicated to preparation, building the knowledge base and local tooling necessary for effective interaction with the HPC environment.
***
### Results

The day's work resulted in a prepared local machine with the correct software and a refreshed understanding of HPC concepts and SLURM commands. However, the critical step of establishing a connection to Amarel was blocked by a network access issue.
***
### Challenges & Pause Points

The primary obstacle is **network connectivity**. To access Amarel from off-campus, a VPN connection to the Rutgers network is required. The recommended software, Cisco AnyConnect, cannot be installed due to a permission error. This prevents the use of both WinSCP and PuTTy, halting progress on file transfer and cluster login.
***
### Questions & Ideas

- **VPN Access:** Is the Cisco AnyConnect software freely available through a different Rutgers portal that I may have missed? Are there any alternative, university-approved VPN solutions for student researchers?
- **Connection Necessity:** Is PuTTy strictly necessary, or can a modern terminal like Git Bash handle the SSH connection directly, potentially simplifying the setup?
- **Access Paths:** Given the VPN issue, should the immediate plan shift to gaining physical access to a campus location with a direct Rutgers network connection, such as Room 330 in the Serin Physics Building?
***
### Next Steps

1. **Resolve Connectivity:** The top priority is to solve the VPN issue. I will search for alternative installation paths for Cisco AnyConnect or contact Rutgers OIT support to resolve the permission error.    
2. **Seek Physical Access:** If the VPN cannot be configured immediately, I will confirm my access privileges to Room 330 or the undergraduate lounge to work directly from a campus machine on the Rutgers network.
3. **Confirm Amarel Access:** Verify that my Amarel account is active and in good standing. If not, I will contact Dr. Keeton for sponsorship.
4. **Execute File Transfer Plan:** Once connected, the immediate technical steps are to create the `GravLensing` subdirectory in my Amarel home directory, transfer the Python scripts via WinSCP, and begin composing the SLURM job script.

##### Tags: #Amarel #HPC #Terminal