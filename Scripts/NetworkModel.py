import tensorflow as tf
from tensorflow import keras
import numpy as np

class GEquivariantDense(tf.keras.layers.Layer):
    """
    Group-equivariant dense layer for 1D parameter vectors
    Applies the same transformation to groups of parameters that should be equivariant
    """
    def __init__(self, units, group_size=4, activation='relu', **kwargs):
        super(GEquivariantDense, self).__init__(**kwargs)
        self.units = units
        self.group_size = group_size
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        # Weight matrix that respects group structure
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.group_size, input_shape[-1] // self.group_size, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
    def call(self, inputs):
        # Reshape input into groups
        batch_size = tf.shape(inputs)[0]
        grouped_inputs = tf.reshape(inputs, [batch_size, self.group_size, -1])
        
        # Apply group-equivariant transformation
        output = tf.einsum('bgi,gio->bo', grouped_inputs, self.kernel)
        output = output + self.bias
        
        return self.activation(output)

def create_gcnn_model():
    """
    G-CNN inspired model for gravitational lensing parameter estimation
    Assumes input features can be grouped (e.g., parameters for multiple images)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),  # 14 input features
        
        # Group features into meaningful groups (e.g., by image or parameter type)
        # Reshape to think in terms of groups - adjust group_size based on your data structure
        tf.keras.layers.Reshape((2, 7)),  # Example: 2 groups of 7 parameters each
        
        # Group-equivariant layer
        tf.keras.layers.Dense(32, activation='relu'),  # Process within groups
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        # Flatten and continue with regular layers
        tf.keras.layers.Flatten(),
        
        # Regular dense layers with regularization
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(12, activation='linear')  # 12 regression outputs
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5),
        loss='mse',
        metrics=['mae']
    )
    return model

def create_advanced_gcnn_model():
    """
    More advanced G-CNN approach using custom grouping
    """
    inputs = tf.keras.layers.Input(shape=(14,))
    
    # Create multiple group representations
    # Group 1: Time delays and positions (assuming first 8 features)
    group1 = tf.keras.layers.Lambda(lambda x: x[:, :8])(inputs)
    group1 = tf.keras.layers.Dense(32, activation='relu')(group1)
    group1 = tf.keras.layers.BatchNormalization()(group1)
    
    # Group 2: Fluxes and other parameters (assuming last 6 features)  
    group2 = tf.keras.layers.Lambda(lambda x: x[:, 8:])(inputs)
    group2 = tf.keras.layers.Dense(32, activation='relu')(group2)
    group2 = tf.keras.layers.BatchNormalization()(group2)
    
    # Combine groups
    combined = tf.keras.layers.Concatenate()([group1, group2])
    
    # Continue with regular architecture
    x = tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    outputs = tf.keras.layers.Dense(12, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5),
        loss='mse',
        metrics=['mae']
    )
    return model

# Keep your existing train_model function - it's perfect!
def train_model(model, x, y):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=64,
        epochs=200,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history