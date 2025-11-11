import tensorflow as tf
from tensorflow import keras

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

def train_model(model, x, y):
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