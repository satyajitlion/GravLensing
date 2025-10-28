### Today's Focus

Today's work focused on analyzing the results for the three mock lens datasets from Amarel (Shear Only, Ellip Only, and Both), classifying the generated mock lenses as being single, double, or quad, and constructing the neural network infrastructure for future data analysis. A key success was confirming the successful generation of mock lenses with the incorporated potential on the Amarel computing cluster.
***
### What I was able to accomplish

- Verified the successful generation of mock lenses with the incorporated potential on Amarel.
    
- Processed and classified lenses from three simulation configurations (Shear Only, Ellip Only, Both) into single, double, quad, and rare cases.
    
- Developed a comprehensive statistical analysis framework to calculate the percentage distribution of each lens type.
    
- Created a modular neural network in `NetworkModel.py` with functions for model creation and training.
    
- Successfully tested the neural network pipeline with synthetic data in the `Analysis.ipynb` notebook to ensure that the code from the script was working properly and that the network was able to load properly into the notebook.
***
### Results

- Successfully classified all mock lenses and calculated the statistical prevalence of singles, doubles, quads, and rare cases for each configuration.
    
- Implemented checks to ensure statistical calculations summed correctly to 100%, validating the analysis.
    
- A functional TensorFlow model was built and tested, featuring a sequential architecture with dense layers and using the Adam optimizer.
***
### Challenges & Pause Points

A notable challenge was handling the "edge cases" of lenses that produced an unusual number of images (not 1, 2, or 4). A system was developed to identify, log, and statistically account for these "rare" lenses, which will be crucial for understanding the completeness of our model and understanding the statistics when it comes to the percentages of mock lenses that produce 1, 2, 3, 4, 5, or greater images (3, 5, and any mock lens that produces >4 images being the "rare" lenses). Furthermore, the code structure currently involves some duplication across the three lensing configurations, presenting an opportunity for future refactoring to create a more efficient and maintainable analysis script.
***
### Questions & Ideas

- How do the percentage distributions of singles, doubles, and quads compare across the three different lensing configurations? 
	- Understanding these differences could reveal the individual and combined effects of shear and ellipticity on image formation. 
- In terms of the "rare" lenses which we are not considering for this project, would this data loss lead to any errors when training the network? 
- For the neural network, the immediate next step is to transition from testing with random data to using the actual lensing parameters. This raises the question of whether the current model architecture is optimal or if a Convolutional Neural Network might be more suitable for analyzing the spatial relationships within the image arrays.
	- Incorporate GCNN using pytorch instead? Will need to look at a pytorch implementation for the Neural Network.
***
### Next Steps

1. Continue studying GCNNs and Group Theory to better understand equivariance and it's relation to invariance and the entire foundation of GCNNs.

2. Work towards replacing the current neural network architecture with the a GCNN. Also consider converting image array dataset into a 2D image such that a CNN could better handle said data. 

3. Replace the neural network's sample data with the actual lensing parameter arrays from the simulations.

4. Perform a comparative analysis of the statistical results across the three configurations to identify key patterns.
    
5. Develop visualizations to effectively communicate the distributions of lens types.

##### Tags: #GravitationalLensing #NeuralNetworks #DataAnalysis #Classification #Python #Keras #TensorFlow




