import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


# actual model itself
model = Sequential([

	# note the input_shape here is completely wrong for my dataset, 
	# this is just an example
	
	Dense(units=16, input_shape=(1,), activation='relu'), # first hidden layer
	Dense(units=32, activation='relu'), # second hidden layer
	Dense(units=2, activation='softmax') # output layer, recall why softmax is used.
])

# note the structure that exists here. I will have to design something similar for my dataset. 


model.summary() 