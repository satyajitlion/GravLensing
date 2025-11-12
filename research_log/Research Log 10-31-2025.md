### Today's Focus

My focus was on starting the practical implementation of a data preprocessing pipeline for a gravitational lensing dataset. The goal was to create a function that standardizes variable-length data from different lensing systems into fixed-size arrays suitable for training a neural network.
***
### What I was able to accomplish

- Began writing the `extract_lens_data` function.
    
- Successfully implemented the logic for padding the image position data (`img_padded`).
    
- Structured the overall function outline, defining the inputs and outputs and setting up the loop to iterate through the dataset.
***
### Results

- Established a clear plan for handling the four key components of the data: image positions, time delays, potentials, and deflections.
    
- Completed the padding and flattening logic for the `img_padded` array. The method of flattening the (x, y) coordinates and then padding the resulting 1D vector was successfully implemented.
    
- The skeleton of the function, including the final `np.concatenate` steps for the input and output vectors, is in place.
***
### Challenges & Pause Points

- Managing the padding for four different data components with different structures (some are 1D per image like `time`, others are 2D like `img` and `deflec`) was really difficult and this is where I mostly had issues working. I tried to ensure I understood the required dimensions for each (`max_images` vs. `max_images * 2`).

- I was unsure of the exact data type and structure of `system['time']` and `system['potent']`, which caused a slight pause in implementing those sections.

- The function started to feel complex with multiple padding steps. I decided to tackle it one component at a time rather than all at once. 
***
### Questions & Ideas

1. Why am I getting so many errors with `.to_value()`? How do I fix these issues to process the data correctly?
    
2. Should I use zero-padding? Would that help in making the arrays properly sized?
    
3. Should we also record the original number of images for each system so the model can ignore the padded zeros?
***
### Next Steps

1. Finish implementing the padding logic for `time_padded`, `potent_padded`, and `deflec_padded`.
    
2. Run the completed function on a small subset of the dataset to check for errors and validate the output array shapes.
    
3. Consider refactoring the repeated padding code into a helper function to make the script more robust and readable.
##### Tags: #DataProcessing #Python #GravitationalLensing #NumPy




