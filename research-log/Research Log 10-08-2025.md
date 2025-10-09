### Today's Focus

Continue neural network regularization research. Improve understanding of dropout and early stopping techniques, explore their practical implementations in Keras, and begin designing neural network architectures suitable for the mock lens data being generated. Bridge theoretical knowledge with practical application by considering how to handle geometric transformations in gravitational lens data.
***
### What I was able to accomplish

- Studyied dropout and early stopping techniques in depth to further deepen my understanding of regularization

- **Developed intuitive understanding of dropout** through real-world analogies and visual representations 

- **Documented practical code examples** for implementing both regularization techniques

- Learned about the **Early stopping implementation in Keras** with proper validation monitoring and patience parameters
***
### Challenges & Pause Points

- Need to figure out how I will apply these regularization techniques specifically to gravitational lens data once it is available.
    
- Still unclear how to ensure neural network predictions remain consistent under translations/rotations of lens systems (hypothetically, it might be sufficient to encode the lens equations as a tensor but I am unsure whether I can do this. Perhaps there is an easier way or a different way to ensure this?)
    
- Still unclear how I should determine optimal dropout rates and early stopping criteria for Grav Lens modeling if I proceed to use this regularization technique.
***
### Questions & Ideas

- How do I apply these regularization techniques specifically to gravitational lens data once available? Which regularization technique do I even use? Would using something like dropout regularization negatively affect the network as the size of the network is decreasing? If the size of the network is decreasing, bias will likely increase as a result. However, the size is only decreasing per layer and is still present for the output layer. 

- How to ensure neural network predictions are invariant to translations and rotations of the lens system?

	- _Idea_: Explore tensor-based approaches where the metric characteristics remain constant despite coordinate changes
    
	- _Idea_: Research geometric deep learning or equivariant neural networks for this specific challenge

- Once data generation is verified, create a data loader that automatically feeds .npy files into Keras

- Could dropout and early stopping work synergistically with other techniques like L2 regularization for our lens data?
***
### Next Steps

1. Track the `mock_lens_100k` job and verify successful generation of all three output files

2. Start working on building the neural network model for mock lens data, incorporating today's regularization learnings. Temporarily use the 30 mock lenses to create the model and ensure that the model can take in much larger datasets.

3. Investigate tensor methods and geometric deep learning for handling coordinate transformations

4. Develop checks to ensure physical consistency of generated lens data

5. Make sure to create some form of algorithm that is able to take in the dictionary I created and is able to unpack the information within said dictionary containing the mock lenses. Prepare for larger simulation runs ($10^5$ lenses) once initial data quality is confirmed. 

6. Add the right-hand side of the following equation to the data dictionary (maybe rerun the Amarel Batch job post edit to incorporate this change as well when generating the mock lenses?)

	```math
	\left\{x_{i}, \Delta t_i\right\}, z_{l}, z_{s} \rightarrow \left\{\phi_{i}, 
	\alpha_{i}\right\}, D_t
	```

##### Tags: #NeuralNetworks #Regularization #Dropout #EarlyStopping #Keras #DeepLearning #Amarel #MockLensGeneration 




