##### Keras and how Neural Networks work: 

- Keras is integrated with TensorFlow
- Numerical data $\rightarrow$ Artificial Neural Network (ANN)
- Standard NN structure:
	Input Layer $\rightarrow$ Hidden Layer $\rightarrow$ Output Layer

##### Forward Propagation (Mathematical structure behind NN layers)

```
\begin{bmatrix} z_{1}\\ z_{2}  \\z_{3} \\ \vdots z_{n}\end{bmatrix} = \begin{bmatrix} w_{11} & w_{12} & \cdots  & w_{1n}\\ w_{21} & w_{22} & \cdots  & w_{2n}\\ w_{31} & w_{32} & \cdots & w_{3n} \\ \vdots & \vdots & \ddots  & \vdots  \end{bmatrix} \cdot \begin{bmatrix} x_{1}\\ x_{2} \\ \vdots \end{bmatrix} + \begin{bmatrix} b_{1}\\ b_{2}  \\b_{3} \\ \vdots \end{bmatrix} \implies Z^{[1]} = W^{[1]}X + b^{[1]}
```

```
\begin{bmatrix} a_{1}\\ a_{2}  \\a_{3} \\ \vdots \end{bmatrix} = \sigma \cdot \begin{bmatrix} z_{1}\\ z_{2}  \\z_{3} \\ \vdots \end{bmatrix} \implies A^{[1]} = \sigma Z^{[1]}
```

- Here, the $Z^{[1]}$ matrix represents the the intermediate neuron values that are calculated using the input layer ($X$) and it's arbitrary set of weights ($W^{[1]}$) and biases ($b^{[1]}$).
- The $A^{[1]}$ matrix represents the activated intermediate neuron values after the activation function, $\sigma$, is applied to $Z^{[1]}$. Following this, $A^{[1]}$ then becomes our new input layer which will be transformed with new weights and biases (via Backward Propagation). 

$$
\begin{bmatrix} z_{21}\\ \vdots \end{bmatrix} = \begin{bmatrix} w_{31} & w_{41} & w_{51} \dots \end{bmatrix} \cdot \begin{bmatrix} a_{11}\\ a_{12} \\ a_{13}\\ \vdots \end{bmatrix} + \begin{bmatrix} b_{1}\\ b_{2}  \\b_{3} \\ \vdots \end{bmatrix} \implies Z^{[2]} = W^{[2]}\cdot A^{[1]} + b^{[2]}
$$
$$
\begin{bmatrix} a_{21} \\ \vdots \end{bmatrix} = \sigma \cdot \begin{bmatrix} z_{21}\\ \vdots \end{bmatrix} \implies A^{[2]} = \sigma Z^{[2]}
$$	
- The activation function used here, $\sigma$, is the sigmoid function. We also need to compute the loss per layer (or error) via the loss function.
##### Loss Function

The loss function is given as 

$$
J = -\frac{1}{m}\sum_{i=1}^{m}L(a^{[2](i)},y^{(i)})
$$
where,

$$
L(a^{[2]},y) = -y\log^{[2]}-1(1-y)\log(1-a^{[2]})
$$

Note, that whenever 

##### Tags: #NeuralNetworks #Keras
