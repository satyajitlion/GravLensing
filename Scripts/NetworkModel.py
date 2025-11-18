import tensorflow as tf
from tensorflow import keras
import numpy as np

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),  # 14 input features
        
        # Layer 1 with regularization
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),  # Add dropout
        tf.keras.layers.BatchNormalization(),  # Add batch norm for stability
        
        # Layer 2 with regularization  
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        # Layer 3 with regularization
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(12, activation='linear')  # 12 regression outputs
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5),  # Lower LR + gradient clipping
        loss='mse',
        metrics=['mae']
    )
    return model

def train_model_with_roto_translation(model, x, y, rotation_prob=0.5, translation_prob=0.5, translation_scale=0.1):
    """
    Training function with both rotation and translation data augmentation
    """
    # Set up callbacks for better training control
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,           # Wait 15 epochs after best validation loss
        restore_best_weights=True,  # Keep the best weights, not the final ones
        verbose=1
    )
    
    # Learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # Reduce LR by half
        patience=10,     # Wait 10 epochs
        min_lr=1e-7,     # Minimum learning rate
        verbose=1
    )
    
    print("Applying rotation and translation augmentation...")
    x_augmented = x.copy()
    
    for i in range(len(x)):
        positions = x[i, :8].reshape(4, 2)
        
        if np.any(positions != 0):
            # ROTATION
            if np.random.random() < rotation_prob:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                positions = positions @ rotation_matrix.T
            
            # TRANSLATION
            if np.random.random() < translation_prob:
                translation = np.random.uniform(-translation_scale, translation_scale, size=2)
                positions = positions + translation
            
            x_augmented[i, :8] = positions.reshape(8)
    
    print("Augmentation complete. Starting training...")
    
    history = model.fit(
        x=x_augmented, 
        y=y,
        validation_split=0.1,
        batch_size=64,    # Increased batch size for stability
        epochs=200,       # Set high but early stopping will cut it short
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]  # Add callbacks
    )
    
    # Print training summary
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history

def train_model_with_rotation(model, x, y, augmentation_prob=0.5):
    """
    Training function with rotation-only data augmentation (backward compatibility)
    """
    return train_model_with_roto_translation(
        model, x, y, 
        rotation_prob=augmentation_prob, 
        translation_prob=0.0  # No translation
    )

def train_model_with_translation(model, x, y, augmentation_prob=0.5):
    """
    Training function with rotation-only data augmentation (backward compatibility)
    """
    return train_model_with_roto_translation(
        model, x, y, 
        rotation_prob=augmentation_prob, 
        translation_prob=0.0  # No translation
    )

def train_model(model, x, y):
    """
    Original training function without augmentation
    """
    # Set up callbacks for better training control
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,           # Wait 15 epochs after best validation loss
        restore_best_weights=True,  # Keep the best weights, not the final ones
        verbose=1
    )
    
    # Learning rate scheduler
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # Reduce LR by half
        patience=10,     # Wait 10 epochs
        min_lr=1e-7,     # Minimum learning rate
        verbose=1
    )
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=64,    # Increased batch size for stability
        epochs=200,       # Set high but early stopping will cut it short
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]  # Add callbacks
    )
    
    # Print training summary
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history