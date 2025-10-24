### Today's Focus

Optimizing and executing the large-scale batch job by implementing a parallel processing strategy using a SLURM job array. A secondary focus was planning for future data preprocessing and model robustness based on advisor feedback.
***
### What I was able to accomplish

- Diagnosed the failure of the original single, long-running batch job due to the 48-hour time limit.
    
- Collaborated with my professor to devise a new strategy: run 10 parallel jobs with a reduced number of iterations (n=10⁴) each, instead of one job with n=10⁵.
    
- Successfully modified the Python script to use the `SLURM_ARRAY_TASK_ID` for generating unique output files, preventing overwrites.
    
- Wrote and configured a new SLURM script utilizing `--array=1-10` to launch the 10 parallel jobs.
    
- Developed and tested a robust Python utility (`combine_results()`) to merge the outputs from all parallel jobs into three final combined arrays.
***
### Results

- The proof-of-concept with 10 mock lenses was successful.
    
- The 10-job SLURM array was submitted and ran successfully.
    
- The file-naming strategy worked perfectly, resulting in 10 sets of three files (e.g., `valShear_1.npy` ... `valShear_10.npy`).
    
- The `combine_results()` function is ready to execute once all jobs are complete.
***
### Challenges & Pause Points

- The main batch job hitting the 48-hour wall-clock limit on the cluster.
    
- Dr. Keeton recommended a "dumb parallel processing" approach to work around the time limit, which was successfully implemented.
    
- Waiting for the 10 parallel jobs to complete their 48-hour run. The process is currently in progress.
    
- How to preprocess our data to be invariant to translations and rotations, which is crucial for a robust model?
***
### Questions & Ideas

- If the jobs were to fail again, then is there a more efficient way to split the work? Should we consider more jobs with even smaller `n`?

- How do Convolutional Neural Networks (CNNs) achieve translation and rotation invariance? What is the core mechanism (e.g., convolutional layers, pooling, data augmentation) and how can we adapt it for our regression task, since we are not building a classifier?

- <u>Idea</u>: Automate the post-processing by having the SLURM script run `combine_results.py` as a dependent job. 
***
### Next Steps

1. **Short-term:**
    - Verify that all 10 jobs in the SLURM array have completed successfully.
        
    - Run the `combine_results.py` script to merge the individual output files.
        
    - Perform a preliminary data quality check on the combined arrays.
        
2. **Medium-term:**
    - **Remind Dr. Keeton** about the Potential Array for the lensing system.
        
    - Begin the planned analysis on the combined dataset.
        
3. **Research & Development:**
    - **Read deeply** into the architectural principles of CNNs, specifically focusing on how convolutional and pooling layers provide translation invariance.
        
    - Investigate how these principles (e.g., using convolutional layers in a regression network) can be applied to our specific problem to make our model robust to rotations and translations in the input data.

##### Tags: #SLURM #HPC #ParallelProcessing #JobArrays #Python #Transformation-Invariance




