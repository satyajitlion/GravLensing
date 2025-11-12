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

- Acquired a clean, validated, and consolidated dataset of 100,000 mock lenses for each model.

- Implemented a more organized and automated data output structure for all future batch jobs.

- Produced the first set of analytical plots (corner plots) from the complete dataset, marking a significant milestone in the project. The analysis is now feasible after fixing the performance issue. 
***
### Challenges & Pause Points

- The initial run of the analysis notebook was prohibitively slow (ran for over an hour without completing). The root cause was traced to verbose printing operations within the loop processing the lenses.

- Paused to diagnose the slow performance. Identified the print statements as the culprit, commented them out, and confirmed the fix led to a massive reduction in processing time. 
***
### Questions & Ideas

- Consider saving the processed data (e.g., calculated potentials, derived parameters) into a new, analysis-ready HDF5 file or similar format to avoid re-running the entire computation every time the notebook is opened. 
	- Could do this for the neural network too to avoid retraining the neural network every single time.

- Now that we have initial corner plots and graphs, what differences can I spot between the previous dataset of 1000 mock lenses per model to this model which has $10^5$ mock lenses per model? How are the corner plots different and what patterns in the corner plots am I realistically trying to spot such that the neural network will also spot those same patterns? 

- Is it worth converting the dataset into a 2D image in order to use the CNN architecture? 
***
### Next Steps

1. Monitor the newly submitted batch job on Amarel to ensure it runs with the updated scripts and improved file structure.
    
2. Use the now-functional `Analysis.ipynb` to perhaps perform a more in-depth analysis of the dataset? Additionally, analyze the data and brainstorm how I can create a pre-processing script to separate the data into pieces that are manageable that could directly be "plugged" into the Neural Network (once it develops).
    
3. Create more specific and informative plots? 
    
4. Start developing the Neural Network and just test out different Network models. Think about if converting the dataset into a 2D image would be worthwhile in the context of which model to use. It might be worth investigating the performance differences between models and comparing which models yield the best results for this type of data.

##### Tags: #DataPreprocessing #TrainingData #Amarel #DeepLearning #GravitationalLensing #MockLensGeneration #NeuralNetworks #Python 




