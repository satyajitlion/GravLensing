### Today's Focus

Revisit the foundational mathematical groundwork for understanding neural networks by reviewing core linear algebra and calculus concepts. Focus on matrix operations, gradient theory, and linear regression as the building blocks for more complex machine learning models. Work towards understanding the need for cost functions and why they are necessarily in a NN (answering questions from 09/24/2025). Begin the review of Logistic Regression.
***
### What I was able to accomplish

- Added clear notes on matrix shapes in NumPy, emphasizing the (rows, columns) convention and its critical importance in network layer calculations.

- Reviewed gradients, connecting the simple derivative $dy/dx$ to the multi-variable gradient $\nabla f$, highlighting its role in pointing in the direction of greatest increase.

- Thoroughly reviewed Linear Regression, breaking it down from simple to multivariable forms.

- Defined the Mean Squared Error (MSE) cost function and explained the intuition behind Gradient Descent for minimizing it, complete with a code snippet for calculating the gradient.

- Initiated the section on Logistic Regression (still need to finish the section).
***
### Results

- Successfully created a concise and clear reference for matrix operations and gradient theory.

- The Linear Regression review is well-documented with both mathematical formulas and practical Python code, reinforcing the connection between theory and implementation.

- The notes are now better structured for quick recall, which will be essential when implementing backpropagation in Neural Networks.
***
### Challenges & Pause Points

- The derivation of the gradient for the MSE cost function, while understood conceptually, can still be a bit tricky to manually execute without error. Need to practice this more.

- The Logistic Regression section is incomplete. The link between its cost function (log loss) and the Linear Regression cost function (MSE) needs to be clarified.
***
### Questions & Ideas

- For Logistic Regression, why can't we use MSE as the cost function? What property of the sigmoid activation function makes log loss more suitable?

- In my NumPy matrix examples, I see that arr1 with shape (1, 3) and arr2 with shape (3, 1) represent different mathematical objects. When we get to the forward propagation equation $z^{[1]} = W^{[1]}X + b^{[1]}$, how do I determine the correct dimensions for $W^{[1]}$ to ensure the matrix multiplication works and produces the output shape needed for the next layer? How would one ensure that the shapes are correctly matched such that matrix/vector multiplication is able to take place?
***
### Next Steps

1. If rooms are accessible, Generate Mock Lenses (priority above all else) and fix errors from Friday. 

2. Finish the notes on Logistic Regression, including its hypothesis (sigmoid), cost function (log loss/BCE), and why it's used for classification.

3. Explicitly link the completed Logistic Regression review to the Neural Network section, showing how a single neuron with a sigmoid activation is essentially a Logistic Regression unit.

4. Use the foundational knowledge from today to re-derive the backpropagation equations in the NN section step-by-step to ensure full understanding.
    
##### Tags: #Foundations #LinearAlgebra #Calculus #GradientDescent




