### Today's Focus

Today's focus was on building the theoretical foundation for future work with Group Equivariant Convolutional Neural Networks (GCNNs). To properly understand these advanced architectures, I dedicated time to studying the fundamental structure and mechanics of standard Convolutional Neural Networks (CNNs).
***
### What I was able to accomplish

- Conducted a comprehensive review of standard CNN architecture and its core principles.
    
- Compiled detailed notes on the function of each layer type: Convolutional, Pooling, Fully Connected, and Dropout.
    
- Broke down the convolutional operation, including the roles of stride and padding.
    
- Studied the purpose and mechanics of key activation functions like ReLU and Sigmoid.
    
- Understood the hierarchical feature learning workflow that makes CNNs so effective for visual data.
***
### Results

- Gained a clear understanding of how CNNs use local receptive fields, weight sharing, and pooling to efficiently process spatial data like images.
    
- Organized the learning into the complete CNN workflow, from input layer through convolutional operations, activation, pooling, and finally to classification via fully connected layers.
    
- This foundational knowledge is a necessary prerequisite for understanding the more complex concept of group equivariance in GCNNs.
***
### Challenges & Pause Points

The primary challenge in this theoretical phase is ensuring a deep, intuitive understanding of why CNNs work, not just how they work. It's easy to memorize the definitions of layers, but the real value comes from understanding how their combination creates a system that is translationally invariant and capable of building a hierarchy of features from simple edges to complex objects. A key pause point was reconciling the mathematical description of the convolution operation with its practical, visual outcome of creating feature maps.
***
### Questions & Ideas

- How exactly does the principle of weight sharing contribute to translational invariance, and how is this concept extended in GCNNs to include other transformations like rotations? 
	- My notes highlight that early layers detect simple features and later layers detect more complex ones; I'm curious about the visual evidence for this and how the depth of the network affects the complexity of the features it can learn. 
- For the project, an idea could be to first build and test a standard CNN on a simplified version of our lensing data to establish a performance baseline. This would make the performance improvement (or specific advantage) of a GCNN much clearer and more quantifiable when we implement it later.
***
### Next Steps

1. Check Amarel/Email to see if the mock lens dataset finished generating.

2. Transition from standard CNNs to studying the paper on Group Equivariant Convolutional Networks (GCNNs).
    
3. Identify the specific limitations of standard CNNs that GCNNs are designed to solve.
    
4. Understand how group theory concepts are integrated into the convolutional layers.
    
5. Begin outlining how a GCNN architecture could be applied to the gravitational lensing dataset.

##### Tags: #GroupTheory #CNN #GroupEquivariantCNNs #Literature #Foundations 




