### Today's Focus

My primary focus was on advancing the project through a discussion with my professor and strengthening my mathematical foundation. The meeting aimed to clarify concepts related to the time delay surface in gravitational lensing and to plan the next steps for implementing a neural network. Following that, I dedicated time to studying group theory to better understand the principles underlying Group Equivariant Convolutional Neural Networks.
***
### What I was able to accomplish

1. Held a meeting to discuss the project's direction, focusing on the physical interpretation of time delays and data preparation for neural networks.
    
2. Dedicated time to studying fundamental group theory, including definitions, properties, and proofs of key lemmas, to build a stronger mathematical foundation for GCNNs.
    
3. Identified concrete next steps for the data processing pipeline and model exploration.
***
### Results

**From Meeting Notes:**

- Clarified that on the time delay surface, images inside the Einstein ring are saddle points, while those outside are local minima.
    
- Understood the arrival order of images: the farthest images (lowest gravitational potential) arrive first, followed by the images inside the Einstein ring.
    
- Key Insight: A transformation that scales the Einstein radius by a factor `w` and the source position by the same `w` will scale the image positions by `w`.
    
- Defined two potential paths for the neural network:
    
    1. Convert lens data into 2D images to train a GCNN.
        
    2. Explore other neural network architectures that may not require 2D pixel data to save memory.
        

**From Group Theory Study:**

- Formally defined a **group** as a set with a binary operation satisfying closure, associativity, identity, and invertibility.
    
- Analyzed a multiplication table example and confirmed it does **not** form a group due to a lack of associativity.
    
- Worked through and understood the proofs for two key lemmas:
    
    - **Lemma 1.2.1:** If a∗a=aa∗a=a, then aa must be the identity element ee.
        
    - **Lemma 1.2.2:** In a group, left inverses are also right inverses, and the identity element is unique.
***
### Challenges & Pause Points

- A key decision point is whether to commit to transforming the data into 2D images. This has significant implications for memory usage and model architecture. Need to weigh the benefits of GCNNs against the computational cost.
    
- Bridging the gap between abstract group theory concepts (e.g., group actions on pixel spaces) and their practical implementation in a neural network is non-trivial. More study may be required to feel comfortable implementing this.
    
- The group theory exercise highlighted that verifying all group properties (especially associativity from a table) can be cumbersome.
***
### Questions & Ideas

***
### Next Steps

##### Tags:




