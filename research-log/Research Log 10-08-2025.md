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


##### Tags:




