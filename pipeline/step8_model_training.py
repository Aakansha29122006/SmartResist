#!/usr/bin/env python
# coding: utf-8
"""
STEPS 8-11 — ANN MODEL LAYER (MLP) + TRAINING
================================================
8.  Feedforward Neural Network (MLP)
    - Fully connected dense layers
    - No embeddings
    - Label encoding only
    - Input: encoded + scaled features
    - Output: susceptibility probability

9.  Training Control (Early Stopping)
    - Stop training if accuracy plateaus for 2 consecutive epochs
    - Stop if accuracy starts decreasing
    - Store training accuracy in a variable for tracking

10. Loss Function — Binary Cross Entropy

11. Imbalance Handling
    - Weighted loss function
    - Balanced sampling
    - Frequency normalization
    - Minimum support filtering

STEP 19 — Model Training Setup
================================
- Batch size: 1024
- Maximum epochs: 100
- Loss: Binary Cross Entropy

STEP 21 — Implementation Constraint
======================================
- Do NOT use PyTorch
- Use TensorFlow/Keras only
"""

import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Feature columns: encoded categoricals + scaled numericals
# Step 7: BioSample_ID NOT included as feature
FEATURE_COLS = [
    'gene_encoded', 'species_encoded', 'drug_encoded',
    'region_start_scaled', 'region_end_scaled', 'region_length_scaled'
]


def build_model(num_genes, num_species, num_drugs):
    """
    Step 8: Build Feedforward ANN (MLP) with Embeddings.
    Uses Embedding layers for categorical variables instead of pure Dense.
    Step 21: TensorFlow/Keras only — NO PyTorch.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Embedding, Concatenate

    tf.keras.utils.set_random_seed(42)

    # Input: [gene_encoded, species_encoded, drug_encoded, start_scaled, end_scaled, length_scaled]
    inputs = Input(shape=(6,), name='combined_input')
    
    gene_input = inputs[:, 0]
    species_input = inputs[:, 1]
    drug_input = inputs[:, 2]
    num_input = inputs[:, 3:]

    gene_emb = Embedding(input_dim=num_genes, output_dim=32)(gene_input)
    species_emb = Embedding(input_dim=num_species, output_dim=16)(species_input)
    drug_emb = Embedding(input_dim=num_drugs, output_dim=16)(drug_input)

    concat = Concatenate()([gene_emb, species_emb, drug_emb, num_input])

    x = Dense(512, activation='relu')(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    # Output: susceptibility probability
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    # Step 10: Binary Cross Entropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def get_feature_arrays(df):
    """
    Extract feature arrays from dataframe.
    Step 7: BioSample_ID used only for grouping — NOT as model feature.
    """
    X = df[FEATURE_COLS].values
    y = df['target'].values
    return X, y


def train_model(train_df, test_df, encoders, scaler, save_dir=None):
    """
    Steps 8-11, 19, 21: Build and train the ANN model.

    Args:
        train_df: Training dataframe with encoded + scaled features.
        test_df: Test dataframe with encoded + scaled features.
        encoders: Dict of label encoders.
        scaler: StandardScaler instance.
        save_dir: Project root directory.

    Returns:
        model: Trained Keras model.
        history: Training history object.
        X_test, y_test: Test arrays for evaluation.
    """
    from sklearn.utils.class_weight import compute_class_weight
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print("\n" + "=" * 60)
    print("STEPS 8-11: ANN MODEL BUILD + TRAINING")
    print("=" * 60)

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.dirname(save_dir)

    ann_dir = os.path.join(save_dir, "ANN_Project")
    model_path = os.path.join(ann_dir, "model.keras")
    encoder_path = os.path.join(ann_dir, "encoders.pkl")
    model_exists = os.path.exists(model_path) and os.path.exists(encoder_path)

    X_train, y_train = get_feature_arrays(train_df)
    X_test, y_test = get_feature_arrays(test_df)

    print(f"  Features: {FEATURE_COLS}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    if model_exists:
        print("\n  Loading pre-trained model...")
        model = tf.keras.models.load_model(model_path)
        print("  ✓ Pre-trained model loaded. Skipping training.")
        return model, None, X_test, y_test

    # Step 11: Compute class weights for imbalance handling (weighted loss)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {
        0: float(weights[0]) if 0 in classes else 1.0,
        1: float(weights[1]) if 1 in classes else 1.0
    }
    print(f"\n  Step 11 — Class weights (imbalance): {class_weight_dict}")

    # Step 8: Build MLP with Embeddings
    num_genes = len(encoders['gene_encoder'].classes_)
    num_species = len(encoders['species_encoder'].classes_)
    num_drugs = len(encoders['drug_encoder'].classes_)
    model = build_model(num_genes, num_species, num_drugs)

    # Step 9: Early stopping
    # Stop if accuracy plateaus for 2 consecutive epochs
    # Stop if accuracy starts decreasing (restore_best_weights=True)
    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=2,
        restore_best_weights=True, mode='max', verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1
    )

    # Step 19: Batch size 1024, up to 100 epochs, BCE loss
    print("\n  Training started (batch=1024, max_epochs=100)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=1024,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Step 9: Store training accuracy in a variable for tracking
    training_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    print(f"\n  Final training accuracy:   {training_accuracy:.4f}")
    print(f"  Final validation accuracy: {val_accuracy:.4f}")

    return model, history, X_test, y_test
