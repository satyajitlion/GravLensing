### Today's Focus

Bridging the gap between the theoretical concepts of group theory and their practical implementation in neural networks. The primary goal was to understand the mathematical formulation of Group Convolutional Neural Networks (G-CNNs) and how they generalize the operation of standard CNNs to achieve equivariance beyond just translation.
***
### What I was able to accomplish

- Solidified the connection between classical CNN convolution and group convolution by analyzing their mathematical definitions.
    
- Understood the key modification in group convolution: the kernel transformation via $g^{-1}y$, which effectively performs "data augmentation on the filters."
    
- Identified the change in domain for feature maps after the first layer: from the pixel grid ($\mathbb{Z}$) to the group space ($G$).
    
- Discovered a practical Python library (`groupy.gconv`) for implementing G-CNNs in PyTorch, specifically for the p4 (rotation) group.
    
- Checked the pytorch code example implementation for GCNNs by Taco Cohen.
***
### Results

- The fundamental difference between standard and group convolution is the transformation of the kernel by the group element $g^{-1}$ before the inner product. This is the mechanism that builds in equivariance.

- A G-CNN's first layer lifts data from the base space ($\mathbb{Z}^{2} \rightarrow G$), and subsequent layers operate entirely within the group space ($G \rightarrow G$).

- The code example concretely shows that a `P4ConvP4` layer outputs a feature map with the shape `(batch, channels, 4 rotations, height, width)`, directly demonstrating the model's internal representation of multiple orientations.
***
### Challenges & Pause Points

- A potential concern about steering too far from the core astrophysics. Am I diving too deep into pure group theory and G-CNN mechanics without a clear, immediate link back to the gravitational lensing problem? The goal is invariance for lensing data, and I need to ensure this foundational work directly serves that aim.

- Translating the compact notation $[f * k](g) = ... k_i(g^{-1}y)$ into an intuitive, visual understanding of how the filter is being transformed and applied.

- Fully grasping the implication that after the first layer, the feature maps themselves are functions on the group $G$, not just on the spatial grid. What does it mean to "convolve on a group"? 
***
### Questions & Ideas

- How do I use a G-CNN to achieve the **invariance** I need for gravitational lensing? Is the final step simply a group-invariant pooling layer (e.g., max-pooling over the group axis) after the last equivariant layer? How does pooling even work?

- How computationally expensive are G-CNNs compared to standard CNNs? The output tensors have an extra group dimension—does this lead to a linear increase in parameters and memory?

- For gravitational lensing, which specific symmetry group is most relevant? Is it the p4 group (90° rotations), a continuous rotation group, or perhaps a group that includes flips (p4m)? 
***
### Next Steps

1. Explicitly map out the pathway from an equivariant G-CNN to an invariant network. The next learning goal is to understand how to add a final invariant layer (e.g., group pooling) to produce a single, transformation-invariant output for lensing parameter prediction.

2. Plan a small-scale experiment to train a G-CNN with an invariant output on a toy problem (e.g., rotated MNIST) and compare it against a baseline.

3. Investigate more modern or well-supported libraries for equivariant deep learning (`escnn`) as potential alternatives to `groupy`. 

4. Formally define the symmetry group for our gravitational lensing data to ensure the chosen G-CNN architecture is appropriate. 

##### Tags: #NeuralNetworks #ENNs #GroupTheory #Transformation-Invariance #GroupEquivariantCNNs #GravitationalLensing 




