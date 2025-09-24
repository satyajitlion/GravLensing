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
\sigma^{\prime}(x) = \left[-1\cdot(1+e^{-x})^{-1-1}\right]\cdot (-e^{-x}) = (1+e^{-x})^{-2}\cdot e^{-x} = \frac{e^{-x}}{(1+e^{-x})^{2}} = \frac{1}{(1+e^{-x})}\cdot\left[\frac{1+e^{-x}-1}{(1+e^{-x})}\right] = \frac{1}{(1+e^{-x})}\cdot\left[1- \frac{1}{1+e^{-x}}\right] = \boxed{\sigma(x)\left[1-\sigma(x)\right]}. 
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
x^(1) → [Layer 1] → [Layer 2] → a^[2](1) → L(1) ┐
x^(2) → [Layer 1] → [Layer 2] → a^[2](2) → L(2) │ → J = average(L(1), L(2), L(3))
x^(3) → [Layer 1] → [Layer 2] → a^[2](3) → L(3) ┘
```
### Backward Propagation (fixing weights and biases using loss function)

- Backpropagation is the algorithm used to train neural networks by calculating the gradient of the loss function with respect to each weight. The process works backwards from the output layer to the input layer.

```math
da^{[2]} = \frac{\partial L}{\partial a^{[2]}} = \frac{\partial}{\partial a^{[2]}}\left(-y\log(a^{[2]})-1(1-y\log(1-a^{[2]}))\right) = \boxed{\frac{-y}{a^{[2]}} + \frac{(1-y)}{1-a^{[2]}}}. 
```

```math
dz^{[2]} = \frac{\partial L}{\partial z^{[2]}} = \frac{\partial L}{\partial a^{[2]}}\times\frac{\partial a^{[2]}}{\partial z^{[2]}} = \left[\frac{-y}{a^{[2]}} + \frac{(1-y)}{1-a^{[2]}} \right]\times \frac{\partial \sigma(z^{[2]})}{\partial z^{[2]}} = \left[\frac{-y(1-a^{[2]}) + (1-y)a^{[2]}}{a^{[2]}(1-a^{[2]})}\right]\times \left[a^{[2]}(1-a^{[2]})\right] = -y + ya^{[2]} + a^{[2]} - ya^{[2]} = \boxed{a^{[2]} - y}.
```

```math
dW^{[2]} = \frac{\partial L}{\partial W^{[2]}} = \frac{\partial L}{\partial z^{[2]}}\times\frac{\partial z^{[2]}}{\partial W^{[2]}} = dz^{[2]} \times \frac{\partial(W^{[2]}a^{[1]}+b^{[2]})}{\partial W^{[2]}} = \boxed{dz^{2}\times a^{[1]} = (a^{[2]} - y) \times a^{[1]}}. 
```

```math
db^{[2]} = \frac{\partial L}{\partial b^{[2]}} = \frac{\partial L}{\partial z^{[2]}}\times\frac{\partial z^{[2]}}{\partial b^{[2]}} = dz^{[2]} \times \frac{\partial(W^{[2]}a^{[1]}+b^{[2]})}{\partial b^{[2]}} = \boxed{dz^{2} = a^{[2]} - y}.
```

```math
da^{[1]} = \frac{\partial L}{\partial a^{[1]}} = \frac{\partial L}{\partial z^{[2]}} \times \frac{\partial z^{[2]}}{\partial a^{[1]}} = dz^{[2]} \times \frac{\partial(W^{[2]}a^{[1]}+b^{[2]})}{\partial a^{[1]}} = \boxed{dz^{[2]}\times W^{[2]} = (a^{[2]} - y)\times W^{[2]}}. 
```

```math
dz^{[1]} = \frac{\partial L}{\partial z^{[1]}} = \frac{\partial L}{\partial a^{[1]}}\times\frac{\partial a^{[1]}}{\partial z^{[1]}} = \left(dz^{[2]}\times W^{[2]}\right)\times \frac{\partial\sigma(z^{[1]})}{\partial z^{[1]}} = \boxed{\left(dz^{[2]}\times W^{[2]}\right)\times\sigma^\prime(z^{[1]}) = \left[(a^{[2]} - y)\times W^{[2]}\right]\times\left(\sigma(z^{[1]})\left[1-\sigma(z^{[1]})\right])\right)}.
```

```math
dW^{[1]} = \frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial z^{[1]}}\times\frac{\partial z^{[1]}}{\partial W^{[1]}} = dz^{[1]}\times \frac{\partial(W^{[1]}X+b^{[1]})}{\partial W^{[1]}} = \boxed{dz^{[1]}\times X = \left\{\left[(a^{[2]} - y)\times W^{[2]}\right]\times\left(\sigma(z^{[1]})\left[1-\sigma(z^{[1]})\right])\right)\right\}\times X}.
```

```math
db^{[1]} = \frac{\partial L}{\partial b^{[1]}} = \frac{\partial L}{\partial z^{[1]}}\times \frac{\partial z^{[1]}}{\partial b^{[1]}} = dz^{[1]}\times \frac{\partial(W^{[1]}X+b^{[1]})}{\partial b^{[1]}} = \boxed{dz^{[1]} = \left(dz^{[2]}\times W^{[2]}\right)\times\sigma^\prime(z^{[1]}) = \left[(a^{[2]} - y)\times W^{[2]}\right]\times\left(\sigma(z^{[1]})\left[1-\sigma(z^{[1]})\right])\right)}.
```

### Gradient Descent & Updating Weights and Biases

- The weight and bias parameters are updated by subtracting the partial derivation of the loss function with respect to those parameters.
- Here α is the learning rate that represents the step size. It controls how much to update the parameter. The value of $\alpha$ is between $0$ to $1$.

```math
W^{[1]} = W^{[1]} - \alpha \cdot dW^{[1]}
```

```math
W^{[2]} = W^{[2]} - \alpha \cdot dW^{[2]}
```

```math
b^{[1]} = b^{[1]} - \alpha \cdot db^{[1]}
```

```math
b^{[2]} = b^{[2]} - \alpha \cdot db^{[2]}
```


### Finding the Optimal Learning Rate $\alpha$
- Learning Rate is one of the most important hyperparameter to tune for Neural network to achieve better performance.  Learning Rate determines the step size at each training iteration while moving toward an optimum of a loss function.
- Keras has a built-in adaptive learning rate that makes it so that the learning rate isn't too small (causing the network to take a long while to converge), and not too big (such that the network doesn't learn at all). 
- It provides the extension of the classical stochastic gradient descent that support adaptive learning rates such as  **Adagrad**, **Adadelta**, **RMSprop** and **Adam**.
	- Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the learning rate.
	- Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients.

### Underfitting, Overfitting, Best-fit

- **Underfit Model** – The model unable to learn the relationship between the input features and target output variable. It works badly on training data and also performs worst on a test dataset. 
- **Overfit Model –** The model has learned the relationship between training dataset’s input features and target output variable very well. It performs best on train dataset but works poor on test data.
- **Bestfit Model –** The model outperforms on both training dataset as well as test dataset.

- Why isn't overfitting good?
	- The overfitted model memorizes the learning patterns between input features and output for the training dataset. Hence, it works best for train dataset but works poor on test unseen dataset.
	- The solution for this problem is to force the network to keep the weight parameters small and prepare the model more generalized. This can be achieved by __regularization__ techniques.
		- There are several regularization methods are used to avoid the overfitting. The regularization techniques make smaller changes to the learning algorithm and prepare model more generalized that even work best on test data.
		- Regularization Examples:
			- L2 Regularization
			- L1 Regularization
			- Dropout
			- Early stopping

### Example (No Keras) Implementation

```python

# In [1]:
# Import required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

# In [2]:
# Read the data
df = pd.read_csv("dataset.csv")
df.shape 
# Out[2]: 
# (200, 3)

# In [3]:
df.head()
# Out[3]:
'''
         x1        x2  target
0  1.065701  1.645795     1.0
1  0.112153  1.005711     1.0
2 -1.469113  0.598036     1.0
3 -1.554499  1.034249     1.0
4 -0.097040 -0.146800     0.0
'''

# In [4]:
# Let's print the distribution of the target variable in class 0 & 1
df['target'].value_counts()
# Out[4]:
'''
0.0    103
1.0     97
Name: target, dtype: int64
'''

# In [5]:
# Let's plot the distribution of the target variable**
plt.scatter(df['x1'], df['x2'], c=df['target'].values.reshape(200,), s=40, cmap=plt.cm.Spectral)
plt.title('Distribution of the target variable')

# Out[5]: 
```

![Alt text](https://studymachinelearning.com/wp-content/uploads/2019/12/plot_target.jpg "Distribution of target var")

```python

# In [6]:
# Let's prepare the data for model training**
X = df[['x1','x2']].values.T
Y = df['target'].values.reshape(1,-1)
X.shape,Y.shape
# Out[6]:
'''
((2, 200), (1, 200))
'''

# In [7]:
m = X.shape[1]             # m - No. of training samples
# Set the hyperparameters
n_x = 2                    # No. of neurons in first layer
n_h = 10                   # No. of neurons in hidden layer
n_y = 1                    # No. of neurons in output layer
num_of_iters = 1000
learning_rate = 0.3

# In [8]:
# Define the sigmoid activation function**
def sigmoid(z):
    return 1/(1 + np.exp(-z))

# In [9]:
# Initialize weigth & bias parameters**
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    parameters = {
        "W1": W1,
        "b1" : b1,
        "W2": W2,
        "b2" : b2
      }
    return parameters

# In [10]:
# Function for forward propagation**
def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
      "A1": A1,
      "A2": A2
     }
    return A2, cache

# In [11]:
# Function to calculate the loss**
def calculate_cost(A2, Y):
    cost = -np.sum(np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2)))/m
    cost = np.squeeze(cost)
    return cost

# In [12]:
# Function for back-propagation**
def backward_prop(X, Y, cache, parameters):
    A1 = cache["A1"]
    A2 = cache["A2"]

    W2 = parameters["W2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m

    grads = {
    "dW1": dW1,
    "db1": db1,
    "dW2": dW2,
    "db2": db2
    }

    return grads

# In [13]:
# Function to update the weigth & bias parameters**
def update_parameters(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    new_parameters = {
    "W1": W1,
    "W2": W2,
    "b1" : b1,
    "b2" : b2
    }

    return new_parameters

# In [14]:
# Define the Model**
def model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate,display_loss=False):
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_of_iters+1):
        a2, cache = forward_prop(X, parameters)

        cost = calculate_cost(a2, Y)

        grads = backward_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, grads, learning_rate)
        
        if display_loss:
            if(i%100 == 0):
                print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return parameters

# In [15]:
trained_parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate,display_loss=True)
# Out[15]:
'''
Cost after iteration# 0: 0.727895
Cost after iteration# 100: 0.438707
Cost after iteration# 200: 0.308236
Cost after iteration# 300: 0.239390
Cost after iteration# 400: 0.200191
Cost after iteration# 500: 0.175058
Cost after iteration# 600: 0.157424
Cost after iteration# 700: 0.144189
Cost after iteration# 800: 0.133626
Cost after iteration# 900: 0.124717
Cost after iteration# 1000: 0.116933
'''

# In [16]:
# Define function for prediction**
def predict(parameters, X):
    A2, cache = forward_prop(X,parameters)
    predictions = A2 > 0.5
    
    return predictions

# In [17]:
# Define function to plot the decision boundary**
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.reshape(200,), cmap=plt.cm.Spectral)

# In [18]:
# Plot the decision boundary**
plot_decision_boundary(lambda x: predict(trained_parameters, x.T), X, Y)

# Out[18]:
```

![Alt text](https://studymachinelearning.com/wp-content/uploads/2019/12/plot_decision_boundary.png "Decision Boundary Graph")


```python
# In [19]:
# Let's see how our Neural Network work with different hidden layer sizes**
plt.figure(figsize=(15, 10))
hidden_layer_sizes = [1, 2, 3, 5, 10,20]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(2, 3, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    
    parameters = model(X, Y, n_x, n_h, n_y, num_of_iters, learning_rate)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    
# Out[19]:
```

![Alt text](https://studymachinelearning.com/wp-content/uploads/2019/12/different_neuron_sizes.png "Different Hidden Layer Sizes")

##### Tags: #NeuralNetworks #Keras