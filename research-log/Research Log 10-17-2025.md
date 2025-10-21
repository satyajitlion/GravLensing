### Today's Focus

Understanding the mathematical fundamentals of group theory and its application to building equivariant and invariant neural networks. The goal was to grasp the core concepts that differentiate these architectures from standard CNNs and to evaluate their potential pros and cons.
***
### What I was able to accomplish

- Defined a mathematical group, including its four essential properties (Closure, Associativity, Identity, Inverse).
    
- Distinguished between a **group** itself and a **group action** on a space (e.g., pixels).
    
- Differentiated the concepts of **equivariance** (`f(t(x)) = t'(f(x))`) and **invariance** (`f(t(x)) = f(x)`).
    
- Analyzed the limitations of standard CNNs, specifically their lack of rotational equivariance.
    
- Compared the theoretical advantages of equivariant networks against the common practice of data augmentation.
***
### Results

- The translational equivariance of CNNs is a specific, limited case of group equivariance. Their fixed filters make them inherently poor at handling rotations not seen in training.
    
- Equivariant NNs embed symmetry constraints directly into the architecture (an inductive bias), leading to benefits like data efficiency and better generalization, but at the cost of expressivity for patterns outside the assumed symmetries.
    
- Data augmentation is an imperfect, post-hoc method to achieve insensitivity to transformations, whereas equivariant models build this property directly into every layer.
***
### Challenges & Pause Points

- Pinpointing the exact mathematical reason why a CNN's convolution operation is _not_ equivariant to rotations. Is it the filter structure, the discrete grid, or both?
    
- Grappling with the practical trade-off between the **data efficiency** of equivariant networks and their **reduced flexibility/expressivity**.
    
- Moving from the abstract definition of a group (`(G, ·)`) to a concrete, intuitive understanding of how it acts on a real-world data space like images.
***
### Questions & Ideas

***
### Next Steps

##### Tags:




