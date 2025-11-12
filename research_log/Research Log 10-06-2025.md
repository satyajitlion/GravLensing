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

- What is the specific Keras syntax for implementing Dropout layers?
    
- In practice, is it better to use a single regularization technique or to combine them (e.g., L2 + Dropout)?
    
- When building the initial neural network model for this project, it might be good to start with ReLU activation in hidden layers (monitoring for signs of "dying" units) and use L2 regularization as a first step to combat overfitting. Also, I should use Softmax on the output layer for classification. I wonder if the Mock Lens data requires classification, however, as it's pretty clear that the data seems to be purely numerical. However, with regard to rotation of the gravitational lens system, I don't want a network that changes the output values simply as a result of rotation or translation of the system. It might be good to think about this for a while.
***
### Next Steps

1. Follow up on the OARC support ticket if no response is received within 1-2 business days.
    
2. Complete the research on regularization by finishing the notes on **Dropout** and **Early Stopping.**

3. Draft the initial architecture of the neural network using Keras, incorporating the insights on activation functions and L2 regularization.
    
4. Implement and test the model on a small subset of data to verify the environment and basic functionality.

##### Tags: #Amarel #Keras #Regularization #ActivationFunctions #NeuralNetworks #Troubleshooting 




