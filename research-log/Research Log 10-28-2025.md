### Today's Focus

Today's work centered on advancing the Gravitational Lens neural network research project. The primary objectives were to analyze the results from three distinct lensing simulations (Shear Only, Ellip Only, and Both), classify the generated mock lenses, and begin constructing the neural network infrastructure for future data analysis. A key success was confirming the successful generation of mock lenses with the incorporated potential on the Amarel computing cluster.
***
### What I was able to accomplish

- Verified the successful generation of mock lenses with the incorporated potential on Amarel.
    
- Processed and classified lenses from three simulation configurations (Shear Only, Ellip Only, Both) into single, double, quad, and rare cases.
    
- Developed a comprehensive statistical analysis framework to calculate the percentage distribution of each lens type.
    
- Created a modular neural network in `NetworkModel.py` with functions for model creation and training.
    
- Successfully tested the neural network pipeline with sample data in the `Analysis.ipynb` notebook.
***
### Results

- Successfully classified all mock lenses and calculated the statistical prevalence of singles, doubles, quads, and rare cases for each configuration.
    
- Implemented checks to ensure statistical calculations summed correctly to 100%, validating the analysis.
    
- A functional TensorFlow model was built and tested, featuring a sequential architecture with dense layers and using the Adam optimizer.
***
### Challenges & Pause Points

A notable challenge was handling the edge cases of lenses that produced an unusual number of images (not 1, 2, or 4). A system was developed to identify, log, and statistically account for these "rare" lenses, which will be crucial for understanding the completeness of our model and the physical regimes where it may break down. Furthermore, the code structure currently involves some duplication across the three lensing configurations, presenting an opportunity for future refactoring to create a more efficient and maintainable analysis script.
***
### Questions & Ideas

***
### Next Steps

##### Tags:




