### Today's Focus

Contact Amarel support at [help@oarc.rutgers.edu](https://mailto:help@oarc.rutgers.edu) to aid with installation help for external packages like shapely on the data science module. For the neural network aspect of the project, learn about the different regularization techniques that are used to avoid overfitting. Learn about the Keras implementation for the neural networks and the ways in which grad descent, nodes, activation functions, optimizers, and regularization is implemented.
***
### What I was able to accomplish

1. Composed and sent an email to OARC support ([help@oarc.rutgers.edu](https://mailto:help@oarc.rutgers.edu)) requesting assistance with installing the `shapely` package on the Amarel cluster.

2. Researched and documented the "Dying ReLU" problem, including its causes and solutions like Leaky ReLU.

3. Compiled a list of common activation functions (Binary Step, Parametric ReLU, ELU, Softmax, Swish, GELU, SELU) with their mathematical definitions and use-cases.

4. Began researching regularization techniques, completing notes on L1 and L2 regularization, including their cost functions and Keras implementations and started learning about the "Dropout" Regularization technique.
***
### Results

- **ReLU vs. Leaky ReLU:** Understood the critical failure mode of the standard ReLU function (Dying ReLU) and how Leaky ReLU provides a robust alternative by preventing zero gradients.
    
- **Activation Function Selection:** Gained a clearer understanding that activation function choice is problem-dependent, with ReLU being a common default for hidden layers and Softmax/Sigmoid being standard for classification output layers.
    
- **L1/L2 Regularization:** Confirmed that these techniques work by adding a penalty on the model's weights directly to the loss function, encouraging simpler models to reduce overfitting. L2 (weight decay) is more commonly used.
***
### Challenges & Pause Points

- The `shapely` package installation on Amarel is currently blocked, pending a response from OARC support.
    
- The research on regularization techniques is incomplete; notes on Dropout and Early Stopping need to be finished.
***
### Questions & Ideas

***
### Next Steps

##### Tags:




