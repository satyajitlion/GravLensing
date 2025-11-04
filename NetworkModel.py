import tensorflow as tf
from tensorflow import keras

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),  # 14 input features
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(12, activation='linear')  # 12 regression outputs
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

def train_model(model, x, y):
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=32,  # Larger batch size for regression
        epochs=100,      # More epochs for regression
        shuffle=True, 
        verbose=1
    )
    return history