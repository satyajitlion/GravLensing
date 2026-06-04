import tensorflow as tf
import numpy as np

def quantile_loss(q, y_true, y_pred):
    """
    Quantile (pinball) loss.
    q=0.5 gives median regression, reduces skew errors.

    q < 0.5 --> targets lower quartile if model is overpredicting
    q > 0.5 --> upper quartile if model is underpredicting (my case)

    https://towardsdatascience.com/quantile-loss-and-quantile-regression-b0689c13f54d/
    """
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))

class MinimumEpochEarlyStopping(tf.keras.callbacks.Callback):
    """Custom early stopping that ensures minimum training epochs"""
    def __init__(self, min_epochs=10, patience=10, monitor='val_loss'):
        super().__init__()
        self.min_epochs = min_epochs
        self.patience = patience
        self.monitor = monitor
        self.best_weights = None
        self.best_epoch = 0
        self.best_value = float('inf')
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        
        if epoch < self.min_epochs:
            # Don't check for improvement in first min_epochs
            return
        
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Restoring weights from epoch {self.best_epoch+1}")
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
    
    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

# ==================== MODEL CREATION FUNCTIONS ====================
''' normal version without that many layers
def create_single_model(loss_type='quantile', quantile=0.5):
    """Model for single image lenses (5 inputs, 3 outputs)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),
        
        tf.keras.layers.Dense(16, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(8, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(3, activation='linear')
    ])
    
    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    return model


def create_double_model(loss_type='quantile', quantile=0.5):
    """Model for double image lenses (8 inputs, 6 outputs)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        
        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(128, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(6, activation='linear')
    ])
    
    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=loss,
        metrics=['mae']
    )
    return model


def create_quad_model(loss_type='quantile', quantile=0.5):
    """Model for quad image lenses (14 inputs, 12 outputs)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),
        
        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(16, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.1),
        
        tf.keras.layers.Dense(12, activation='linear')
    ])
    
    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    return model'''

def create_single_model(loss_type='mse', quantile=0.6):
    """Model for single image lenses (5 inputs, 3 outputs)
    Architecture change: 4 hidden layers instead of 2 I believe."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(5,)),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(16),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(3, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # slightly lower lr for deeper net
        loss=loss,
        metrics=['mae']
    )
    return model


def create_double_model(loss_type='mse', quantile=0.6):
    """Model for double image lenses (8 inputs, 6 outputs)
    Architecture Change: 5 hidden layers, wider in the middle."""
    
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(6, activation='linear')
        
    ])
    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # lower lr for deeper network
        loss=loss,
        metrics=['mae']
    )
    return model '''

    """Minimal double model with 2 hidden layers (64)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        
        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(6, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # tweak with lr --> 0.1 to 0.001
        loss=loss,
        metrics=['mae']
    )
    return model


def create_quad_model(loss_type='mse', quantile=0.6):
    """Model for quad image lenses (14 inputs, 12 outputs)
    Architecture change: 4 hidden layers, wider than the original."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(14,)),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),

        tf.keras.layers.Dense(12, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=loss,
        metrics=['mae']
    )
    return model

# ==================== TRAINING FUNCTIONS (AUGMENTATION DISABLED) ====================

def train_model_with_roto_translation_singles(model, x, y, rotation_prob=0.3, translation_prob=0.3, translation_scale=0.1):
    """Training function for singles (augmentation disabled for now)"""
    if x.shape[1] != 5:
        raise ValueError(f"Single model expects 5 input features, got {x.shape[1]}")
    
    early_stopping = MinimumEpochEarlyStopping(min_epochs=10, patience=10, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    # ---- Augmentation disabled ----
    # print(f"Applying rotation and translation augmentation for singles (prob={rotation_prob})...")
    # x_augmented = x.copy()
    # for i in range(len(x)):
    #     positions = x[i, :2].reshape(1, 2)
    #     if np.any(positions != 0):
    #         if np.random.random() < rotation_prob:
    #             angle = np.random.uniform(0, 2 * np.pi)
    #             cos_a, sin_a = np.cos(angle), np.sin(angle)
    #             rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    #             positions = positions @ rotation_matrix.T
    #         if np.random.random() < translation_prob:
    #             translation = np.random.uniform(-translation_scale, translation_scale, size=2)
    #             positions = positions + translation
    #         x_augmented[i, :2] = positions.reshape(2)
    # print("Augmentation complete. Starting training...")
    # history = model.fit(x=x_augmented, y=y, validation_split=0.1, batch_size=16, epochs=100, shuffle=True, verbose=1, callbacks=[early_stopping, reduce_lr])
    # ---------------------------------
    
    # Use original x (no augmentation)
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=16,
        epochs=100,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    
    return history

def train_model_with_roto_translation_doubles(model, x, y, rotation_prob=0.5, translation_prob=0.5, translation_scale=0.1):
    """Training function for doubles (augmentation disabled for now)"""
    if x.shape[1] != 8:
        raise ValueError(f"Double model expects 8 input features, got {x.shape[1]}")
    
    early_stopping = MinimumEpochEarlyStopping(min_epochs=5, patience=15, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    
    # ---- Augmentation disabled ----
    # print("Applying rotation and translation augmentation for doubles...")
    # x_augmented = x.copy()
    # for i in range(len(x)):
    #     positions = x[i, :4].reshape(2, 2)
    #     if np.any(positions != 0):
    #         if np.random.random() < rotation_prob:
    #             angle = np.random.uniform(0, 2 * np.pi)
    #             cos_a, sin_a = np.cos(angle), np.sin(angle)
    #             rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    #             positions = positions @ rotation_matrix.T
    #         if np.random.random() < translation_prob:
    #             translation = np.random.uniform(-translation_scale, translation_scale, size=2)
    #             positions = positions + translation
    #         x_augmented[i, :4] = positions.reshape(4)
    # print("Augmentation complete. Starting training...")
    # history = model.fit(x=x_augmented, y=y, validation_split=0.1, batch_size=64, epochs=200, shuffle=True, verbose=1, callbacks=[early_stopping, reduce_lr])
    # ---------------------------------
    
    # Use original x (no augmentation)
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

def train_model_with_roto_translation_quads(model, x, y, rotation_prob=0.3, translation_prob=0.3, translation_scale=0.1):
    """Training function for quads (augmentation disabled for now)"""
    if x.shape[1] != 14:
        raise ValueError(f"Quad model expects 14 input features, got {x.shape[1]}")
    
    early_stopping = MinimumEpochEarlyStopping(min_epochs=10, patience=10, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    
    # ---- Augmentation disabled ----
    # print(f"Applying rotation and translation augmentation for quads (prob={rotation_prob})...")
    # x_augmented = x.copy()
    # for i in range(len(x)):
    #     positions = x[i, :8].reshape(4, 2)
    #     if np.any(positions != 0):
    #         if np.random.random() < rotation_prob:
    #             angle = np.random.uniform(0, 2 * np.pi)
    #             cos_a, sin_a = np.cos(angle), np.sin(angle)
    #             rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    #             positions = positions @ rotation_matrix.T
    #         if np.random.random() < translation_prob:
    #             translation = np.random.uniform(-translation_scale, translation_scale, size=2)
    #             positions = positions + translation
    #         x_augmented[i, :8] = positions.reshape(8)
    # print("Augmentation complete. Starting training...")
    # history = model.fit(x=x_augmented, y=y, validation_split=0.1, batch_size=32, epochs=100, shuffle=True, verbose=1, callbacks=[early_stopping, reduce_lr])
    # ---------------------------------
    
    # Use original x (no augmentation)
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=32,
        epochs=100,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    
    return history

# ==================== SIMPLE TRAINING FUNCTIONS ====================

def train_simple_model(model, x, y, model_type='single'):
    """Simple training without augmentation for debugging"""
    print(f"\nTraining {model_type} model without augmentation...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    if model_type == 'single':
        batch_size = 16
    elif model_type == 'quad':
        batch_size = 32
    else:
        batch_size = 64
    
    history = model.fit(
        x=x,
        y=y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=100,
        shuffle=True,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    return history

def train_with_cross_validation(model_creator, x_data, y_data, model_type='single', n_splits=3):
    """Train with cross-validation for small datasets"""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data), 1):
        print(f"\nFold {fold}/{n_splits}:")
        
        x_train_fold, x_val_fold = x_data[train_idx], x_data[val_idx]
        y_train_fold, y_val_fold = y_data[train_idx], y_data[val_idx]
        
        model = model_creator()
        
        if model_type == 'single':
            history = train_simple_model(model, x_train_fold, y_train_fold, 'single')
        elif model_type == 'quad':
            history = train_simple_model(model, x_train_fold, y_train_fold, 'quad')
        else:
            history = train_model_with_roto_translation_doubles(model, x_train_fold, y_train_fold)
        
        histories.append(history)
    
    return histories

# ==================== ANALYSIS FUNCTIONS ====================

def analyze_training_history(history, model_name):
    """Analyze if model actually learned"""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = len(train_loss)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    print(f"Epochs trained: {epochs}")
    print(f"Initial training loss: {train_loss[0]:.4f}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"Improvement: {((train_loss[0] - train_loss[-1]) / train_loss[0]) * 100:.1f}%")
    
    if epochs <= 1:
        print("Problem: Model stopped at epoch 1!")
        print("The model hasn't learned anything.")
        return False
    
    if (train_loss[0] - train_loss[-1]) / train_loss[0] < 0.1:
        print("Problem: Minimal improvement (<10%)")
        print("Model may not be learning effectively.")
        return False
    
    print("Model shows significant learning")
    return True

def compare_models_performance(models_dict, test_data_dict):
    """Compare multiple models' performance"""
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    results = []
    for name, (model, x_test, y_test) in models_dict.items():
        if len(x_test) > 0:
            loss, mae = model.evaluate(x_test, y_test, verbose=0)
            results.append({
                'Model': name,
                'Test Loss': loss,
                'Test MAE': mae,
                'Samples': len(x_test)
            })
    
    results.sort(key=lambda x: x['Test MAE'])
    
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['Model']}:")
        print(f"   MAE: {r['Test MAE']:.4f}, Loss: {r['Test Loss']:.4f}, Samples: {r['Samples']}")
    
    return results

# ==================== BACKWARD COMPATIBILITY ====================

def train_model_with_rotation(model, x, y, augmentation_prob=0.5):
    """Backward compatibility (augmentation disabled)"""
    # Calls the appropriate function but with rotation prob 0 (augmentation disabled internally anyway)
    if x.shape[1] == 14:
        return train_model_with_roto_translation_quads(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 8:
        return train_model_with_roto_translation_doubles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 5:
        return train_model_with_roto_translation_singles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    else:
        raise ValueError(f"Unknown input shape: {x.shape[1]}")

def train_model_with_translation(model, x, y, augmentation_prob=0.5):
    """Backward compatibility (augmentation disabled)"""
    if x.shape[1] == 14:
        return train_model_with_roto_translation_quads(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 8:
        return train_model_with_roto_translation_doubles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    elif x.shape[1] == 5:
        return train_model_with_roto_translation_singles(model, x, y, rotation_prob=0.0, translation_prob=0.0)
    else:
        raise ValueError(f"Unknown input shape: {x.shape[1]}")

def train_model(model, x, y):
    """Original training function without augmentation"""
    early_stopping = MinimumEpochEarlyStopping(min_epochs=5, patience=100, monitor='val_loss') # original patience = 15.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
    
    if x.shape[1] == 5:
        batch_size = 16
    elif x.shape[1] == 8:
        batch_size = 64
    else:
        batch_size = 32
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=1000, # changes from 200 for testing
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history

def create_model():
    """Legacy function - creates quad model."""
    print("Warning: create_model() creates a quad model. Use specific functions instead.")
    return create_quad_model()

#--------------simpler models without augmentation -------------------#

def create_simple_double_model(loss_type='mse', quantile=0.6):
    """Minimal double model with 2 hidden layers (64)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        
        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.0),
        
        tf.keras.layers.Dense(6, activation='linear')
    ])

    if loss_type == 'quantile':
        loss = lambda yt, yp: quantile_loss(quantile, yt, yp)
    else:
        loss = 'mse'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['mae']
    )
    return model

def train_simple_model(model, x, y):
    """Original training function without augmentation"""
    early_stopping = MinimumEpochEarlyStopping(min_epochs=5, patience=15, monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-10, verbose=1)
    
    if x.shape[1] == 5:
        batch_size = 16
    elif x.shape[1] == 8:
        batch_size = 64
    else:
        batch_size = 32
    
    history = model.fit(
        x=x, 
        y=y,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=1000,
        shuffle=True, 
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )
    
    print(f"\n=== TRAINING SUMMARY ===")
    print(f"Stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    
    return history