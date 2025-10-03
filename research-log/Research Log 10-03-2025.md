### Today's Focus

Understanding different types of activation functions in neural networks, their mathematical formulations, visual characteristics, advantages, and limitations - with particular attention to the vanishing gradient problem.
***
### What I was able to accomplish

- Studied and documented three main activation functions: Sigmoid, Tanh, and ReLU
- Compiled mathematical formulas for each activation function
- Understood the graphical representations and output ranges of each function
- Analyzed the specific problems associated with each activation function
- Gained a comprehensive understanding of the vanishing gradient problem through analogies
***
### Results

**Key Findings:**

- **Sigmoid**: Used for binary classification output layers, but suffers from vanishing gradients, slow convergence, and non-zero-centered output (range: 0-1)
    
- **Tanh**: Zero-centered improvement over sigmoid (range: -1 to 1), but still has vanishing gradient issues
    
- **ReLU**: Computationally efficient, avoids vanishing gradient problem, uses simple max(0,z) operation

**Note**: I discovered that as an activation function, ReLU is generally preferred over sigmoid and tanh due to its computational efficiency and ability to mitigate the vanishing gradient problem.
***
### Questions & Ideas

- Why exactly does ReLU avoid the vanishing gradient problem while sigmoid and tanh don't?
- Are there situations where sigmoid or tanh might still be preferable to ReLU?
- What about the "dying ReLU" problem I've heard mentioned?
- How do other activation functions (Leaky ReLU, ELU, Swish) compare to these three?
***
### Next Steps

1. Research and understand the "dying ReLU" problem and its solutions
2. Explore other modern activation functions beyond these three basic types
3. Get Started on Keras and learn how models are implemented through Keras
4. Fix Mock Lens Generation issues with Amarel (contact Amarel Support, touch up on this issue with Dr. Keeton)
5. Try to implement a small-scale example neural network model with Keras to test how model implementation works.

##### Tags: #NeuralNetworks #ActivationFunctions #Foundations 




