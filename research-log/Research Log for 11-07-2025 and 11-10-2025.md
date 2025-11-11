### The Focus for these Two Days

Understanding the mathematical foundation and implementation of Group Equivariant Convolutional Neural Networks (GCNNs), specifically for the $\text{SE}(2)$ group (roto-translations in 2D). The goal is to bridge the theoretical understanding of equivariant cross-correlations with their potential application of coding GCNNs in order to analyze lens image data, where rotation and translation invariance are important.
***
### What I was able to accomplish

- I completed lecture 1.3 on GCNNs, where I focused on understanding how standard CNNs incorporate SE(2)-equivariant math to be roto-translationally equivariant.
    
- Simplified the mathematical formulas for the following operations:
    
    1. **Lifting Correlation**: Transforms a 2D input feature map into a 3D feature map on the $\text{SE}(2)$ group by correlating with rotated and translated kernels.
        
    2. **Group Correlation**: Operates directly on 3D feature maps, performing template matching in the full group space.
        
- Outside of the lecture content, I did some external reading to be able to understand the lecture properly as some of the concepts were hard to visualize.
    
- I also created detailed notes with equations and an provided intuitive examples to myself for some of the concepts, such as that of the "template matching" process to help me understand and piece together how GCNNs function.
***
### Results

- I found that GCNNs generalize CNNs by making the feature maps functions on a group (e.g., $\text{SE}(2)$), not just on spatial coordinates. A lifting convolution outputs a 3D feature map and explicitly accounts for the orientation of the features.
    
- The equivariant cross-correlation can be broken down into rotation and translation operations on the kernel.
    
- The primary difference between Lifting and Group correlations is their input domain: Lifting takes 2D to 3D, while Group correlation operates from 3D to 3D.
    
- GCNNs not only detect features but also their _relative poses_ (locations and orientations), leading to more structured and interpretable features. Invariance can be achieved at the end via pooling over the orientation axis.
***
### Challenges & Pause Points

- The notation involving lifts and inner products in spaces was very confusing. It required a lot of reading for me to continue the lecture to understand visually what was going on. The notation is still quite unfamiliar, however, it is starting to make more sense as I read more about GCNNs.
    
- Visualizing the transition from a 2D feature map to a 3D "group space" was challenging. The diagram was crucial for building intuition.
    
- Distinguishing the output of the Lifting correlation from the Group correlation required careful re-reading. The latter also rotates the kernel relative to the orientation of the _input feature_ itself.
***
### Questions & Ideas
    
- Are there standard PyTorch or TensorFlow libraries (e.g., `escnn`, `keops`) that implement these $\text{SE}(2)$ lifting and group convolutions, or would they need to be coded from scratch?
***
### Next Steps

1. Identify a suitable GCNN library and attempt to create the appropriate network model for the current dataset.
    
2. Decide whether to use lifting convolutions, group convolutions, and where to pool for invariance.
    
3. Search for existing papers applying GCNNs for astrophysical datasets (if there are any), particularly those involving symmetric or rotationally variant features, to learn from their design choices.

##### Tags: #GCNN #Equivariance #SE2 #LiftingConvolution #GroupConvolution #GravitationalLensing #Research #Theory




