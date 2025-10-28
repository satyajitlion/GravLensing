import tensorflow as tf
from tensorflow import keras

# create model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    # use adam optimizer
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# train the model
def train_model(model, x, y):
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=10, 
        epochs=30, 
        shuffle=True, 
        verbose=2
    )
    return history