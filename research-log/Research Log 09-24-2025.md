### Today's Focus

Todays' focus was different from yesterday as I did not have access key to the rooms. Therefore, with the VPN aspect keeping mock lens generation on hold, I focused on the neural network side and on studying how mathematically the layers are structured in a standard neural network that is hand coded. 

***
### What I was able to accomplish

I created a document called [Keras and Neural Networks](Notes/Keras and Neural Networks.md) to serve as notes as I learn more about Neural Networks and the mathematics behind the layers. In particular, I spent today analyzing how the inputs and the hidden layers are transformed by the inclusion of weights and biases to get a desired output. This, I analyzed for a standard ANN and I worked out the math accordingly. This involved starting from the input layers, multiplying them by weight matrices and adding biase matrices. Then, through back propagation and gradient descent, one can also adjust the weights and biases again to find their desired output. However, this process produces something called loss, which is a measure of how poorly the model's predictions align with the true, desired outcomes for a given input. Following this, I then documented the structure of the Network including how loss function's work (singular input sample vs multiple input samples). I additionally learned about why finding the optimal value for $\alpha$, or the learning rate hyperparameter, is important. In the document I linked above, I went over Keras's built in options for yielding the optimal value for the learning rate and went over underfitting, overfitting, and best-fit models. Their importance was also discussed, in addition to why overfitting should not be overlooked and the regularization methods one can use to circumvent overfitting. Finally, I went through a sample implementation of a neural network using these principles and numpy alone (using the studymachinelearning.com website). The process proved to be quite tedious in terms of model creation, hinting at how simple an implementation via Keras might be.
***

### Questions & Ideas

My questions at the moment are, how does a standard ANN differ from a CNN or an RNN and in the context of my project, which one might be most beneficial to code? I also should consider Dr. Keeton's suggestion on how to implement a model such that rotations will not change my answer for said model... meaning that as input, the x and y directions I currently have for the sources need to be tweaked in order to form sort of a metric such that when a transformation such as a rotation acts on the system, the system isn't changed. For instance, if I were to rotate or move the gravitational lens, the Network should be able to see that a simple transformation took place that didnt' change the gravitational lens itself but only its position/orientation.
***
### Next Steps

Watch the following video [Keras With TensorFlow](https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=423s) and continue taking notes on Neural Networks if it's not possible to access the room 330/undergrad lounge on campus tomorrow. If I can access either room 330 or the undergrad lounge, work on mock lens generation instead. 


##### Tags: #Keras #NeuralNetworks #TrainingData




