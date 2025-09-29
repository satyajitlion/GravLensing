### Today's Focus

Dedicate time to understanding the theoretical foundations of neural networks and understand how Keras works. The goal was to understand the mathematics behind NNs by deriving the core equations for a standard, hand-coded Artificial Neural Network (ANN), focusing on forward propagation, loss functions, and the principles of backpropagation.
***
### What I was able to accomplish

I made significant progress in building a fundamental understanding of neural network mechanics. The primary output was the creation of detailed notes in the document [Keras and the Math Behind Neural Networks](https://github.com/satyajitlion/GravLensing/blob/72f7b39408a4cbb90cab4cf252d79171e64333a4/Notes/Keras%20and%20Neural%20Networks.md) My work involved:
- **Mathematical Derivation:** I systematically worked through the mathematical transformations that occur in an ANN, starting from the input layer through to the hidden layers. This involved deriving how input data is transformed via weight matrices and bias vectors to produce an output.
- **Core Concepts Documented:** I documented key concepts including the role of loss functions for both individual samples and batches, the critical importance of the learning rate ($\alpha$) as a hyperparameter, and the challenges of model fitting (underfitting, overfitting, and the ideal best-fit).    
- **Practical Implementation:** To solidify my understanding, I followed a tutorial to implement a basic neural network using only NumPy, without the abstraction of a high-level framework like Keras. This hands-on exercise highlighted the complexity involved in manual implementation and underscored the utility of libraries like Keras for streamlining the process.

This deep dive provided a strong conceptual foundation that will be crucial for effectively designing and troubleshooting neural networks later in the project.
***
### Questions & Ideas

- **Architectural Difference:** How do Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) fundamentally differ from the standard ANN I studied today? Which architecture is most suitable for analyzing 2D gravitational lensing images?
- **Invariance in the Model:** As suggested by Dr. Keeton, a major consideration is how to design the network's input or structure to be invariant to rotations and translations. The model should recognize that a rotated lens is the same underlying system. What is the best way to engineer this? Should the input data be pre-processed into a rotation-invariant metric, or should the network architecture itself (perhaps using a CNN) inherently handle these symmetries?
***
### Next Steps

1. **Continue Keras/TensorFlow Learning:** If physical access to campus workstations is not possible, the next step is to watch the tutorial video [Keras With TensorFlow](https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=423s) to transition from theoretical math to practical implementation.
2. **Prioritize Mock Lens Generation:** If I can access Room 330 or the undergraduate lounge, the immediate priority is to resolve the Python dependency issues on Amarel and begin the large-scale generation of mock lenses, as this data is a prerequisite for training any neural network.
    
##### Tags: #Keras #NeuralNetworks #TrainingData



