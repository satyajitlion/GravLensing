### Today's Focus

My focus was to successfully train the first neural network model on the preprocessed gravitational lensing data. This involved integrating the data pipeline with the model architecture, executing the training process, and analyzing the results to validate our entire approach.
***
### What I was able to accomplish

- Successfully diagnosed and fixed the `.to_value()` errors that were halting progress by discovering `system['time']` was already numeric.
    
- Finalized the `extract_lens_data` function and processed the entire `vals_shear` dataset.
    
- Integrated the data pipeline with the neural network model, creating a complete training workflow.
    
- Successfully trained the model for 100 epochs and achieved excellent convergence.
    
- Created visualization plots to analyze the training history and model performance.
    
- Updated the model architecture to match the exact dimensions of our dataset (14 inputs, 12 outputs).
***
### Results

- Established a fully functional ML pipeline from raw data to trained model.
    
- The model converged effectively, with final metrics showing strong performance:
    - **Training Loss (MSE):** 3.87e-04
    - **Training MAE:** 0.0118
    - **Validation Loss (MSE):** 3.63e-04
    - **Validation MAE:** 0.0106
        
- Confirmed the input shape (14 features) and output shape (12 values) match our data structure.
    
- The close alignment between training and validation curves indicates the model is generalizing well without overfitting.
    
- Created loss and MAE plots that clearly show stable and consistent improvement throughout training.
***
### Challenges & Pause Points

- The main initial challenge was debugging the `.to_value()` error, which was resolved by inspecting the actual data structure.
    
- Determining the correct input dimension (14) required calculating: `max_images * 2` for images + `max_images` for times + 2 for redshifts = `4*2 + 4 + 2 = 14`.
    
- Ensuring the output dimension (12) matched: `max_images` for potentials + `max_images * 2` for deflections = `4 + 4*2 = 12`.
    
- The training progression showed some minor fluctuations around epochs 7-11, but the model quickly stabilized and continued improving.
***
### Questions & Ideas

- The model is learning effectively, but the very low final MAE (0.011) raises questions - are we potentially overfitting on a simple pattern? Should we test on a held-out test set?

- Now that I have a working baseline model, it is important to do the following:
	- Evaluate on a completely unseen test set to verify real-world performance
	- Experiment with more complex architectures (deeper networks, skip connections)
	- Consider normalizing the input data to potentially improve convergence speed

- Given the "success" of this simple feedforward network (as in it works functionally), I could experiment with implementing the Group Equivariant CNN which might work better potentially.
***
### Next Steps

1. Save the trained model and evaluate it on a separate test dataset to confirm generalization capability.

2. Analyze which output variables (potentials vs. deflections) the model predicts best and where it struggles.

3. Begin experimenting with different model architectures and hyperparameters to push performance further.

4. Implement and test feature normalization to see if it improves training efficiency.

5. Document the complete workflow for future reproducibility and scaling. Fix naming errors and try to keep everything consistent in the GitHub documentation.  

##### Tags: #ModelTraining #NeuralNetwork #GravitationalLensing #Keras #TensorFlow 




