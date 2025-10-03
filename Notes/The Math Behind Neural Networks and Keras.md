### Quick Matrix Note:

```python
# Input
from numpy import array
arr1 = array([[10,25,15]])
arr2 = array([[10],[25],[15]])
print("shape of arr1 :",arr1.shape)
print("shape of arr2 :",arr2.shape)

'''
Output:
shape of arr1 : (1, 3)
shape of arr2 : (3, 1)
'''
```

- As can be seen above, the shape is given as (row, column) where an array $[1,2,3]$ has a shape of (1,3) for 1 row and 3 columns. Comparing this to the following array: 
```math
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
```
   which has a shape of (3,1). This is the same as the python script above which is why the shape indicated above yields (1,3) and (3,1) respectively and is vital to understand for the latter parts of the math required for the NN.
### Quick Gradient Note:

Recall that for a derivative with respect to any specific variable, let's say we take the derivative of $y$ with respect to $x$ such that we have $\frac{dy}{dx}$, then this accounts for the change along the $x$ direction for the $y$ function. Extending this to 3 dimensions, recall that we take the gradient of a function with respect to it's multiple variables to find the **change** in that direction for the function. 

For example, f(x,y,z) can have three partials as such:

```math
\begin{aligned}
\frac{\partial}{\partial x}\left[f(x,y,z)\right] \\ 
\frac{\partial}{\partial y}\left[f(x,y,z)\right] \\
\frac{\partial}{\partial z}\left[f(x,y,z)\right]
\end{aligned}
```

which yield to the following gradient:

```math
\begin{aligned}
\nabla f(x,y,z) = 
	\begin{bmatrix}
	   \partial_{x}f \\
	   \partial_{y}f \\
	   \partial_{z}f
	\end{bmatrix}
\end{aligned}
```

Therefore, like the one dimensional example, the gradient gives a vector that encodes the change in the $x$, $y$, and $z$ directions for a function.

***
### Linear Regression Review

The linear regression model learns to understand the relationship between input variables and the output variable. This is the backbone of Machine Learning as a whole.  

##### 1. Simple Regression

```math
y = mx+b
```

- Single input and single output 
- Here, the input parameter ($x$) is scaled by some scalar coefficient ($m$) and added to by some intercept or <u>bias</u> coefficient ($b$) to produce the output ($y$), which is our prediction.
##### 2. Multivariable Regression

```math
f(x,y,z) = w_{1}x + w_{2}y + w_{3}z
```

- When there are multiple input variables, we need multiple scalar coefficients called <u>weights</u> such that they may produced our desired output or prediction.  
##### 3. Cost Function

The cost function finds the **best** optimized value for the weight coefficients which helps in finding the line of best fit for the data points we have. This function is the <u>error rate between the observation's actual target value and the predicted target value</u>. The difference between the actual target value and predicted one is called <u>error</u>.

- **MEAN SQUARED ERROR (MSE)**
	- This is a type of cost function that measures the averaged difference between the actual target values and the predicted target values. The main goal here is to minimize the error in order to improve the accuracy of the model. 

```math 
\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (\text{pred}_i - y_i)^2
```
 
##### 4. Grad. Descent

Gradient Descent is a method of updating the weight and bias coefficients iteratively in order to minimize the MSE by calculating the gradient of the cost function. Recall that taking the gradient of a function returns a vector output that encodes it's changes in x, y, z and etc directions. By taking the gradient of the cost function, we are trying to learn how it changes in the directions of it's inputs such that we can minimize those changes to effectively minimize the error output given by the cost function. This will help us alter the weights and biases accordingly to perform such a minimization for the cost function. In more general terms, it will lead to the algorithm learning the nuanced relationships between input and output parameters. 

Example:

```python
import numpy as np

pred_y = m * x + b
cost_function = (y - pred_y)**2

# Cost function as f(m, b)
def f(m, b):
    return (y - (m*x + b))**2

# Partial derivatives 
def df_dm(m, b):
    return -2 * x * (y - (m*x + b))

def df_db(m, b):
    return -2 * (y - (m*x + b))

# Gradient vector
gradient = np.array([df_dm(m, b), df_db(m, b)])
```

***
### Logistic Regression Review

- Logistic regression is used when the target variable is categorical. It is a classification algorithm used to assign a sample to a specific class. 
- The main difference between linear regression and logistic regression is that linear regression predicts the continuous value where Logistic regression apply the sigmoid function to its output and return a probability value and later it mapped to discrete class.

To more concretely explain this, consider an example for linear vs. logistic regression:

1. Linear Regression:
	- Predict the sale price of the house
	- Predict student’s exam score
	- Predict share market’s movements
2. Logistic Regression:
	- Classify ticket type such as first-class, second class, etc.
	- Predict the sentiment of the text.
	- Classify season such as winter, summer, monsoon.

The "activation function" called Sigmoid is used to map predicted value to probabilities between 0 and 1 in logistic regression. This function looks like a s shape as such:
<p align="center">
  <img src="https://studymachinelearning.com/wp-content/uploads/2019/09/sigmoid_graph.png" />
</p>

where $\sigma(z) = \left(1+e^{-z}\right)^{-1}$ and $\sigma(z)$ is the output from 0 to 1 (probability estimate) and $z$ is the <u>input function</u> (i.e z = mx+b).

##### Predicting a Target Variable
- Logistic regression returns the probability of the test samples being positive
- If probability is close to 1, the model is more confident that the test sample is in class 1.
- Ex: Suppose that we have a patient's thyroid test report. If the model returns 0.85 as an output, that means that the patient is thyroid positive with an 85% chance. If it returns 0.3, that means the patient only has 30% chance of being thyroid positive.

##### Hypothesis representation of Linear Regression

```math
h_{\theta}(x) = mx + b
```

##### Hypothesis representation of Logistic Regression

```math
h_{\theta}(x) = \sigma(mx+b) = \frac{1}{1+e^{mx+b}}
```

##### Cost Function in Logistic Regression:

- The cost function represents optimization objective. 
- The Cross-Entropy function is used as a cost function in Logistic Regression (also known as the Log Loss Function).
- The cross-entropy cost function can be divided into two separate cost function for class 1 and class 0.

```math
\text{Cost}(h\Theta(x),y) = 
\begin{cases}
-\log(h_{\theta}(x)) & \text{if } y = 1 \\
-\log(1-h_{\theta}(x)) & \text{if } y = 0
\end{cases}
```

And so, the above can be composed into one function as such:

```math
J(\theta) = \frac{1}{m}\sum\left[y^{(i)}\log(h_{\theta}(x(i))) + \left(1-y^{(i)}\right)\log(1 - h_{\theta}(x(i)))\right]
```

Note that here, if $y = 0$, the first part of the summation naturally cancels out.. If $y = 1$, the second part of the summation naturally cancels out.

##### Revisiting grad descent again for logistic regression:
- Gradient descent is a method of updating the weight coefficient iteratively in order to minimize the cost function by calculating the gradient of it.
- Gradient descent in Logistic Regression works similarly as Linear Regression, the difference is only the hypothesis function. However, the derivative of the Logistic regression is complicated. To minimize the cost function, we need to apply gradient descent function on each parameter.

##### Mapping probabilities to classes

- Logistic Regression returns the probabilities between 0 and 1. Therefore, we need to select a specific threshold value to map probability to discrete class.
- For instance, let us select the threshold value as 0.5. Then, the probability being greater than or equal to 0.5 leads to class 1 and the probability being less than 0.5 lead to class 0. In other words, if P represents the probability, then $P \geq 0.5$ means it's in class 1 and $P < 0.5$ means it's in class 0.
- Example:
	Suppose that our logistic model returned 0.2 probability for predicting cancer. We would then classify this observation as a negative class..
```math
\text{Probability Map to Discrete Class} = 
\begin{cases}
p \geq 0.5,  & \text{class = 1}  \\
p < 0.5,  & \text{class = 0} 
\end{cases}
```

***
### Intro to Keras and how Neural Networks work: 

- Keras is integrated with TensorFlow
- Numerical data $\rightarrow$ Artificial Neural Network (ANN)
- Standard NN structure:
	Input Layer $\rightarrow$ Hidden Layer $\rightarrow$ Output Layer
#### Forward Propagation (Mathematical structure behind NN layers)

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
#### Sigmoid Activation Function

```math
\sigma(x) = \frac{1}{1+e^{-x}} = (1+e^{-x})^{-1}
```

Note here that

```math
\sigma^{\prime}(x) = \left[-1\cdot(1+e^{-x})^{-1-1}\right]\cdot (-e^{-x}) = (1+e^{-x})^{-2}\cdot e^{-x} = \frac{e^{-x}}{(1+e^{-x})^{2}} = \frac{1}{(1+e^{-x})}\cdot\left[\frac{1+e^{-x}-1}{(1+e^{-x})}\right] = \frac{1}{(1+e^{-x})}\cdot\left[1- \frac{1}{1+e^{-x}}\right] = \boxed{\sigma(x)\left[1-\sigma(x)\right]}. 
```

#### Loss Function

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
#### Backward Propagation (fixing weights and biases using loss function)

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

#### Gradient Descent & Updating Weights and Biases

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


#### Finding the Optimal Learning Rate $\alpha$
- Learning Rate is one of the most important hyperparameter to tune for Neural network to achieve better performance.  Learning Rate determines the step size at each training iteration while moving toward an optimum of a loss function.
- Keras has a built-in adaptive learning rate that makes it so that the learning rate isn't too small (causing the network to take a long while to converge), and not too big (such that the network doesn't learn at all). 
- It provides the extension of the classical stochastic gradient descent that support adaptive learning rates such as  **Adagrad**, **Adadelta**, **RMSprop** and **Adam**.
	- Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to how frequently a parameter gets updated during training. The more updates a parameter receives, the smaller the learning rate.
	- Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients.

#### Underfitting, Overfitting, Best-fit

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

#### Example (No Keras) Implementation

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

<p align="center">
  <img src="https://studymachinelearning.com/wp-content/uploads/2019/12/plot_target.jpg" />
</p>

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
<p align="center">
  <img src="https://studymachinelearning.com/wp-content/uploads/2019/12/plot_decision_boundary.png" />
</p>

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
<p align="center">
  <img src="https://studymachinelearning.com/wp-content/uploads/2019/12/different_neuron_sizes.png" />
</p>

#### Activation Functions and Different types:

- The activation function of a node defines the output of that node given an input or set of inputs in the neural network. 
- The activation function allows the neural network to learn a non-linear pattern between inputs and target output variable.
- A neural network without an activation function is just like a linear regression model which is not able to learn the complex non-linear pattern. Therefore, the activation function is a key part of the neural network. 

##### Types of Activation Functions:

1. Sigmoid - covered already; **This is generally used for the output layer in binary classification problem.**
	```math
		\sigma(x) = f(x) = \frac{1}{1 + e^{-x}}
	```

	Problems with using the Sigmoid Function:
	- “vanishing gradients” problem occur
	- Slow convergence
	- Sigmoids saturate and kill gradients.
	- Its output isn’t zero centred. It makes the gradient updates go too far in different directions. 0 < output < 1, and it makes optimization harder.

<p align="center">
  <img src="https://studymachinelearning.com/wp-content/uploads/2019/10/sigoid_plot.png" />
</p>

2. Tanh - hyperbolic tangent

	```math
		\tanh(z) = f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
	```

	Similar (in terms of the graph) to the Sigmoid function with the difference that the Tanh function is zero-centered instead of being centered at some constant value. Therefore, the Tanh non-linearity is always preferred to the sigmoid nonlinearity. The range of output value is between -1 to 1. 

	Problems with using the Tanh Function:
	- same “vanishing gradients” problem occurs
<p align="center">
  <img src="https://studymachinelearning.com/wp-content/uploads/2019/10/tanh_plot.jpg" />
</p>



3. ReLU - rectified linear unit 


#### Vanishing Gradient Problem explained:

Let us look at an example of a ball on top of a mountain. If you put a ball on the top of a mountain, it will roll downhill. This is the case until it hits a valley. In optimization, this concept is called "gradient descent" as discussed earlier. You pick a point and then move in the direction of the steepest slope (gradient) until you find a valley. The valley is, ideally, the best way to do something.

Another example would be if your goal is to train an image recognition algorithm, you then change a bunch of parameters and calculate the accuracy of recognizing some example images (loss function). Then you select the change that gives you the best result (the gradient), discard the rest and do this again, until you can no longer get any better results.

The <u>vanishing gradient problem</u> means that tweaking the parameters doesn't change anything. In the example of the ball rolling down a hill, it would be the ball getting stuck on a frozen lake instead of a valley. This is problematic because you don't know if there's a hole somewhere in the frozen lake, or if it's really as good as it's ever going to be.

This is problem is related to another problem with gradient descent: **the local optimum**. This is what happens if the ball comes to a stop in a valley on top of the mountain, rather than getting all the way to the lowest point.

For example if you have an image recognition algorithm, it corresponds to a "model" (meaning: a bunch of numbers that are used to recognize things in images) which gives you the most accurate image recognition.

***
### Tags: #NeuralNetworks #Keras