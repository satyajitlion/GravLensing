### Intro to Keras and how Neural Networks work: 

- Keras is integrated with TensorFlow
- Numerical data $\rightarrow$ Artificial Neural Network (ANN)
- Standard NN structure:
	Input Layer $\rightarrow$ Hidden Layer $\rightarrow$ Output Layer

### Forward Propagation (Mathematical structure behind NN layers)

```math
\begin{bmatrix} z_{1}\\ z_{2}  \\z_{3} \\ \vdots \\ z_{n}\end{bmatrix} = \begin{bmatrix} w_{11} & w_{12} & \cdots  & w_{1n}\\ w_{21} & w_{22} & \cdots  & w_{2n}\\ w_{31} & w_{32} & \cdots & w_{3n} \\ \vdots & \vdots & \ddots  & \vdots  \end{bmatrix} \cdot \begin{bmatrix} x_{1}\\ x_{2} \\ \vdots \\ x_{n}\end{bmatrix} + \begin{bmatrix} b_{1}\\ b_{2}  \\b_{3} \\ \vdots \\ b_{n}\end{bmatrix} \implies z^{[1]} = W^{[1]}X + b^{[1]}
```

```math
\begin{bmatrix} a_{1}\\ a_{2}  \\a_{3} \\ \vdots \\ a_{n}\end{bmatrix} = \sigma \left( \begin{bmatrix} z_{1}\\ z_{2}  \\z_{3} \\ \vdots \\ z_{n}\end{bmatrix} \right) \implies a^{[1]} = \sigma(z^{[1]})
```

- Here, the $Z^{[1]}$ matrix represents the intermediate neuron values that are calculated using the input layer ($X$) and its arbitrary set of weights ($W^{[1]}$) and biases ($b^{[1]}$).
- The $A^{[1]}$ matrix represents the activated intermediate neuron values after the activation function, $\sigma$, is applied to $Z^{[1]}$. Following this, $A^{[1]}$ then becomes our new input layer which will be transformed with new weights and biases (via Backward Propagation).

```math
\begin{bmatrix} z_{21}\\ \vdots \\ z_{2m}\end{bmatrix} = \begin{bmatrix} w_{31} & w_{41} & w_{51} & \cdots \end{bmatrix} \cdot \begin{bmatrix} a_{11}\\ a_{12} \\ a_{13}\\ \vdots \\ a_{1n}\end{bmatrix} + \begin{bmatrix} b_{1}\\ b_{2}  \\b_{3} \\ \vdots \\ b_{m}\end{bmatrix} \implies z^{[2]} = W^{[2]}\cdot a^{[1]} + b^{[2]}
```

```math
\begin{bmatrix} a_{21} \\ \vdots \\ a_{2m}\end{bmatrix} = \sigma \left( \begin{bmatrix} z_{21}\\ \vdots \\ z_{2m}\end{bmatrix} \right) \implies a^{[2]} = \sigma(z^{[2]})
```

- The activation function used here, $\sigma$, is the sigmoid function. We also need to compute the loss per layer (or error) via the loss function.

### Sigmoid Activation Function

```math
\sigma(x) = \frac{1}{1+e^{-x}} = (1+e^{-x})^{-1}
```

Note here that

```math
\sigma^{\prime}(x) = \left[-1\cdot(1+e^{-x})^{-1-1}\right]\cdot (-e^{-x}) = (1+e^{-x})^{-2}\cdot e^{-x} = \frac{e^{-x}}{(1+e^{-x})^{2}} = \frac{1}{(1+e^{-x})}\cdot\left[\frac{1+e^{-x}-1}{(1+e^{-x})}\right] = \frac{1}{(1+e^{-x})}\cdot\left[1- \frac{1}{1+e^{-x}}\right] = \boxed{\sigma(x)\left[1-\sigma(x)\right]} 
```

### Loss Function

The cost function (loss function for the entire model) is given as

```math
J = -\frac{1}{m}\sum_{i=1}^{m}L(a^{[2](i)},y^{(i)})
```

where the loss function $L$ (for a single training sample) is given as follows:

```math
L(a^{[2]},y) = -y\log(a^{[2]}) - (1-y)\log(1-a^{[2]})
```

Note, that whenever $y = 1$, $L(a^{[2]},1) = -\log(a^{[2]})$, and when $y = 0$, $L(a^{[2]},0) = -\log(1-a^{[2]})$.

Here is how one would visualize the above errors:

```
Single Input x^(i) → [Layer 1] → [Layer 2] → Output a^[2](i) → Loss L(i)
```

```
TRAINING SET (m samples)
│
├─ Sample 1: x^(1) → [Layer 1] → [Layer 2] → a^[2](1) → L(1)
├─ Sample 2: x^(2) → [Layer 1] → [Layer 2] → a^[2](2) → L(2) 
├─ Sample 3: x^(3) → [Layer 1] → [Layer 2] → a^[2](3) → L(3)
└─ ...
└─ Sample m: x^(m) → [Layer 1] → [Layer 2] → a^[2](m) → L(m)
│
↓
J = (L(1) + L(2) + L(3) + ... + L(m)) / m
```
### Backward Propagation (fixing weights and biases using loss function)

- Backpropagation is the algorithm used to train neural networks by calculating the gradient of the loss function with respect to each weight. The process works backwards from the output layer to the input layer.

```math
da^{[2]} = \frac{\partial L}{\partial a^{[2]}} = \frac{\partial}{\partial a^{[2]}}\left(-y\log(a^{[2]})-1(1-y\log(1-a^{[2]}))\right) = \boxed{\frac{-y}{a^{[2]}} + \frac{(1-y)}{1-a^{[2]}}} 
```

```math
dz^{[2]} = \frac{\partial L}{\partial z^{[2]}} = \frac{\partial L}{\partial a^{[2]}}\cdot\frac{\partial a^{[2]}}{\partial z^{[2]}} = \left[\frac{-y}{a^{[2]}} + \frac{(1-y)}{1-a^{[2]}} \right]\cdot \frac{\partial \sigma(z^{[2]})}{\partial z^{[2]}} = \left[\frac{-y(1-a^{[2]}) + (1-y)a^{[2]}}{a^{[2]}(1-a^{[2]})}\right]\cdot \left[a^{[2]}(1-a^{[2]})\right] = -y + ya^{[2]} + a^{[2]} - ya^{[2]} = \boxed{a^{[2]} - y}
```

```math
dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]T}
```

```math
db^{[2]} = \frac{1}{m}\sum_{i=1}^{m}dZ^{[2](i)}
```

```math
dZ^{[1]} = W^{[2]T}dZ^{[2]} \cdot \sigma'(Z^{[1]})
```

```math
dW^{[1]} = \frac{1}{m}dZ^{[1]}X^T
```

```math
db^{[1]} = \frac{1}{m}\sum_{i=1}^{m}dZ^{[1](i)}
```

### Gradient Descent & Updating Weights and Biases

Using 
##### Tags: #NeuralNetworks #Keras