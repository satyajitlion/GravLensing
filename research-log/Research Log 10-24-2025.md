### Today's Focus

The primary focus was on data management, validation, and initiating the first full-scale analysis of the complete dataset. The day involved transitioning from data preparation to initial exploratory data analysis.
***
### What I was able to accomplish

- Successfully confirmed that the Amarel batch job ran correctly. Retrieved the output files and verified their integrity and format.

- Executed the `combine_arrays.py` script locally, successfully merging the 10 split files for each model (`valShear`, `valEllip`, `valBoth`) into three single arrays of $10^5$ mock lenses each.

- Pipeline Optimization:
    - Edited the Slurm script to automatically create a dedicated output folder for better file organization in future runs.
        
    - Updated and re-uploaded all scripts to Amarel to include the new potential parameter in the data dictionary.
        
    - Initiated a new batch job with the improved scripts.

- Ran the `Analysis.ipynb` notebook on the full 100,000-lens dataset. Identified and resolved a major performance bottleneck by commenting out code that was printing each mock lens individually. Successfully generated corner plots and other graphs from the dataset. 
***
### Results

***
### Challenges & Pause Points

***
### Questions & Ideas

***
### Next Steps

##### Tags:




