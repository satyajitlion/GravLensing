### Today's Focus

Initial exploration of group-equivariant convolutional neural networks (G-CNNs) for gravitational lensing analysis. The goal was to use learned G-CNN theory and identify potential implementations that could handle the rotational symmetries present in gravitational lensing systems, particularly for processing multiple images of the same lens system. For this, I used the [Jack Ding's repository](https://github.com/diningeachox/G-CNN). 
***
### What I was able to accomplish

- Researched the theoretical foundation of G-CNNs from Cohen & Welling's 2016 paper, focusing on how they encode transformation equivariance directly into network architecture.
    
- Identified the original 2016 GrouPy implementation (TensorFlow/Chainer) and a more recent 2023 PyTorch implementation as potential starting points.
    
- Analyzed how P4 and P4M group symmetries (90° rotations and reflections) could map to gravitational lensing scenarios where multiple images represent the same source under different transformations.
    
- Set up initial development environment with Python 3.12 and began dependency installation for testing both implementations.
***
### Results

- Confirmed that G-CNNs theoretically align well with gravitational lensing problems, where the same physical source appears in multiple orientations due to lensing geometry.
    
- Determined that the PyTorch implementation would be more maintainable long-term due to better ecosystem support.
    
- Established that building equivariance directly into convolution weights rather than data augmentation could provide significant efficiency and generalization benefits for lensing analysis.
***
### Challenges & Pause Points

- The original 2016 implementation relies on deprecated frameworks (Chainer) and old CUDA versions, making it impractical for current development.
    
- Understanding the mathematical formulation of group convolutions required significant background reading in group theory and representation theory.
    
- Uncertainty about how to map abstract group theory concepts (P4, P4M) to the specific symmetries present in gravitational lensing systems.
***
### Questions & Ideas

- Could G-CNNs naturally handle the fact that lensed images are not exact rotations but related through the lens equation? Would this require custom group definitions?
    
- How would the network architecture need to be modified to process variable numbers of images per lens system (1-4 images)?
    
- Would the rotational equivariance help the network learn lens model parameters more efficiently by explicitly encoding the relationship between multiple images of the same source?
***
### Next Steps

1. Download and attempt to run the PyTorch G-CNN implementation on standard benchmarks (rotated MNIST) to verify basic functionality.
    
2. Begin designing how gravitational lensing data (time delays, image positions, fluxes) could be formatted as input to G-CNN layers.
    
3. Research if custom group structures beyond P4/P4M would be needed for gravitational lensing symmetries.
    
4. Plan architecture modifications to handle the specific data structure of gravitational lensing systems.

##### Tags: #GroupTheory #GCNN #CNN #Debugging #Troubleshooting #NeuralNetwork #GroupEquivariantCNNs 




