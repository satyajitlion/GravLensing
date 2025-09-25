### Today's Focus

Get access to room 330 and try to log into Amarel using the Bash terminal. If able to login, create a sub-directory named "GravLensing". Copy current python scripts, ```pygravlens.py```, ```constants.py```, and ```generateMockLens.py``` into the subdirectory using WinSCP. Schedule the batch job via SLURM and figure out how creating jobs works in Amarel. 
***
### What I was able to accomplish

I was able to get into room 330 and access Amarel. I logged into Amarel via ssh and my Git Bash terminal. I found out that I didn't need to install PuTTy to access Amarel. Following this, I copied the necessary files to my home directory in Amarel using WinSCP and changed the ```num_mock``` parameter to $10^5$. I additionally created a sub-directory named "GravLensing," where I shifted all of the files (the files that I had mentioned in the focus section). I then went back to my computational astrophysics and the OARC resources for how to create SLURM files and how scheduling batch jobs works with Amarel. This entire process took me hours of work as I kept stumbling into errors when creating the SLURM file in terms of how to create and format it. I was additionally missing some  of the resources that were listed in my computational astrophysics course as they seemed to have been deleted or archived. After consulting google, the OARC's official website, I was finally able to  create and run my ```run_lens_job.slurm``` file. Running the ```sbatch run_lens_job.slurm``` command, I ran into various errors that stemmed from not having installed the required python libraries properly within my directory (such as numpy, astropy, scipy, shapely, and etc.). I am still stuck on this problem and will have to work on fixing it tomorrow.
***
### Results

Accessed room 330 in Serin Physics Building. Logged onto Amarel. Copied required contents into my directory for Amarel. Created a SLURM file for scheduling the batch job. Allocated appropriate resources for task (24 hours, 8 Gigs, cluster=Amarel, etc.). 
***
### Problems

Problems with installing appropriate libraries via the terminal on Amarel. The job isn't able to run as it fails when trying to access the specific libraries I imported within my python script such as NumPy, Matplotlib, etc. I also had issues using the right python version (there were 2, the 2.7.5 version and the 3.8.2 version). My directory kept on defaulting to the 2.7.5 version and I had to find the path of the 3.8.2 python version to properly load it via the ```Module Load``` command.
***
### Next Steps

Figure out a way to load the libraries properly into Amarel. Consult Dr. Burkhart or OARC on this? Make sure to not use the login node to install anything as that can lead to fatal errors from what the OARC stated on their website. Following this, temporarily work on the neural networks/keras aspect of the project while waiting for the mock lenses to be generated  (studying how a Neural Network is implemented using Keras this type). Find the differences between ANN, RNN, CNN and analyze which one would be more suitable for this research.

##### Tags: #Amarel #High-PerformanceComputing #MockLens #MockLensGeneration #Terminal 




