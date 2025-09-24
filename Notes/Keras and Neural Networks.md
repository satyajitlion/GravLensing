### Notes: 
- Keras is integrated with TensorFlow
- Numerical data $\rightarrow$ Artificial Neural Network (ANN)
- Standard NN structure:
	Input Layer $\rightarrow$ Hidden Layer $\rightarrow$ Output Layer
- Forward Propagation (Hidden Layer math)

	$\begin{bmatrix} z_{1}\\ z_{2}  \\z_{3} \\ \vdots \end{bmatrix} = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22}  \\ w_{31} & w_{32}  \\ \vdots  & \vdots \end{bmatrix} \cdot \begin{bmatrix} x_{1}\\ x_{2} \\ \vdots \end{bmatrix} + \begin{bmatrix} b_{1}\\ b_{2}  \\b_{3} \\ \vdots \end{bmatrix} \implies Z^{[1]} = W^{[1]}X + b^{[1]}$ 
	
	$\begin{bmatrix} a_{1}\\ a_{2}  \\a_{3} \\ \vdots \end{bmatrix} = \sigma \cdot \begin{bmatrix} z_{1}\\ z_{2}  \\z_{3} \\ \vdots \end{bmatrix} \implies A^{[1]} = \sigma Z^{[1]}$	

	- Here, the $Z^{[1]}$ matrix represents the the intermediate neuron values that are calculated using the input layer ($X$) and it's arbitrary set of weights ($W^{[1]}$) and biases ($b^{[1]}$).
- 
	

##### Tags: #NeuralNetworks #Keras
