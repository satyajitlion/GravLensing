##### Keras and how Neural Networks work: 

- Keras is integrated with TensorFlow
- Numerical data $\rightarrow$ Artificial Neural Network (ANN)
- Standard NN structure:
	Input Layer $\rightarrow$ Hidden Layer $\rightarrow$ Output Layer

##### Forward Propagation (Mathematical structure behind NN layers)

$$
Z^{[1]} = \left[ \begin{array}{cccc}
w_{11} & w_{12} & \cdots & w_{1n} \\
w_{21} & w_{22} & \cdots & w_{2n} \\
w_{31} & w_{32} & \cdots & w_{3n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{n1} & w_{n2} & \cdots & w_{nn}
\end{array} \right]
\left[ \begin{array}{c}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{array} \right]
+ \left[ \begin{array}{c}
b_1 \\ b_2 \\ b_3 \\ \vdots \\ b_n
\end{array} \right]
\implies Z^{[1]} = W^{[1]}X + b^{[1]}
$$	

$$
A^{[1]} = \sigma\left( \left[ \begin{array}{c}
z_1 \\ z_2 \\ z_3 \\ \vdots \\ z_n
\end{array} \right] \right)
\implies A^{[1]} = \sigma(Z^{[1]})
$$

- Here, the $Z^{[1]}$ matrix represents the intermediate neuron values that are calculated using the input layer ($X$) and its arbitrary set of weights ($W^{[1]}$) and biases ($b^{[1]}$).
- The $A^{[1]}$ matrix represents the activated intermediate neuron values after the activation function, $\sigma$, is applied to $Z^{[1]}$. Following this, $A^{[1]}$ then becomes our new input layer which will be transformed with new weights and biases (via Backward Propagation). 

$$
Z^{[2]} = \left[ \begin{array}{cccc}
w_{31} & w_{41} & w_{51} & \cdots \\
w_{32} & w_{42} & w_{52} & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{array} \right]
\left[ \begin{array}{c}
a_{11} \\ a_{12} \\ a_{13} \\ \vdots \\ a_{1n}
\end{array} \right]
+ \left[ \begin{array}{c}
b_1 \\ b_2 \\ b_3 \\ \vdots \\ b_m
\end{array} \right]
\implies Z^{[2]} = W^{[2]} \cdot A^{[1]} + b^{[2]}
$$

$$
A^{[2]} = \sigma\left( \left[ \begin{array}{c}
z_{21} \\ z_{22} \\ \vdots \\ z_{2m}
\end{array} \right] \right)
\implies A^{[2]} = \sigma(Z^{[2]})
$$	

- The activation function used here, $\sigma$, is the sigmoid function. We also need to compute the loss per layer (or error) via the loss function.

##### Loss Function

The loss function is given as 

$$
J = -\frac{1}{m}\sum_{i=1}^{m} L(a^{[2](i)}, y^{(i)})
$$

where,

$$
L(a^{[2]}, y) = -y \log(a^{[2]}) - (1-y) \log(1-a^{[2]})
$$

Note, that whenever $y = 1$, $L(a^{[2]}, 1) = -\log(a^{[2]})$, and when $y = 0$, $L(a^{[2]}, 0) = -\log(1-a^{[2]})$.

##### Tags: #NeuralNetworks #Keras