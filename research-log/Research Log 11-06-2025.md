### Today's Focus

My focus was to begin Lecture 1.3, aiming to understand how the theoretical concepts of group theory translate into the practical operation of a group-equivariant cross-correlation. I sought to grasp the mathematical formulation that generalizes the standard convolution to the $\text{SE}(2)$ group.
***
### What I was able to accomplish

- Started Lecture 1.3 of the "Group Equivariant Deep Learning" series.
    
- Reviewed the standard cross-correlation operation and connected it to the left-regular representation of the translation group.
    
- Understood the interpretation of cross-correlation as a form of template matching, where a kernel is matched against a signal under all translations.
    
- Began learning about the mathematical representation of the $\text{SE}(2)$-equivariant cross-correlation, which involves both rotational and translational parts of the left-regular representation.
***
### Results

- I can better see the connection between the abstract group representation theory from Lecture 1.2 and the concrete convolution operation used in neural networks. The left-regular representation is the mechanism that "moves" the kernel.
    
- The standard CNN cross-correlation is revealed as a special case that is equivariant only to the translation group.
    
- For the generalized $\text{SE}(2)$ cross-correlation, one part of the representation handles the rotation of the kernel and another part handles its translation. This provides a better intuition for how to achieve roto-translational equivariance. The math is frustrating due to it's notation at times but having a physical understanding / image of the network's underlying math is quite helpful.
***
### Challenges & Pause Points

- The notation in the $\text{SE}(2)$ cross-correlation formula is something that is taking me some time as the notation I am not entirely familiar with. It took me a while to unpack the meaning of the symbols and I had to revisit lecture 1.2 to recall certain concepts.
    
- Visualizing how this is efficiently computed across a full feature map and for multiple orientations is still challenging for me. The computational aspect has not yet been covered either and I'm simply studying math which is tad concerning as my goal was to be able to implement this network immediately. However, this has been delayed a bit due to how conceptually heavy and nuanced this topic is.
***
### Questions & Ideas

- Since images are made of pixels and we can't have infinite rotations, how is this group convolution calculated in practice? What are the actual steps?

- Does this mean the output feature map of an $\text{SE}(2)$ convolution layer has a third dimension for orientation, in addition to height and width? 
***
### Next Steps

1. Finish watching Lecture 1.3 to see the full derivation and any practical insights or examples provided.
    
2. After finishing the lecture, actively search for a code snippet or a better diagram that illustrates the implementation of a $\text{SE}(2)$ convolution layer.
    
3. Create the GCNN network model and replace the current network model I was using to test my data. 

4. Check to see if the input provided to the network is in the appropriate formatting for GCNNs. Formatting the input might be different for GCNNs as opposed to a normal fully connected network. 

##### Tags: #GCNN #SE2 #GroupConvolution #Equivariance #GroupTheory #ResearchLog 




