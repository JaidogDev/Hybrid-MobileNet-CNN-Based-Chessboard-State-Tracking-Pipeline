#!/usr/bin/env python3
"""
Retrain the model with balanced dataset and improved architecture
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import json

def create_improved_model(num_classes=13, img_size=96):
    """Create improved CNN model with better architecture"""
    
    model = tf.keras.Sequential([
        # Data augmentation layer (built into model)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        
        # Improved CNN architecture
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(img_size, img_size, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_balanced_dataset(data_dir, img_size=96, test_split=0.2):
    """Load dataset with proper train/validation split"""
    from tensorflow.keras.utils import image_dataset_from_directory
    
    data_dir = Path(data_dir)
    
    # Create train/val datasets
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=test_split,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=32
    )
    
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=test_split,
        subset="validation", 
        seed=123,
        image_size=(img_size, img_size),
        batch_size=32
    )
    
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def compute_class_weights(data_dir):
    """Compute class weights for imbalanced dataset"""
    
    class_counts = {}
    data_path = Path(data_dir)
    
    for class_dir in data_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] = count
    
    print("Class distribution:")
    total = sum(class_counts.values())
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:>8}: {count:>5} ({count/total*100:>5.1f}%)")
    
    # Calculate balanced weights
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    weights = compute_class_weight(
        'balanced',
        classes=np.arange(len(classes)),
        y=np.repeat(np.arange(len(classes)), counts)
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(weights)}
    
    print("\nComputed class weights:")
    for i, (cls, weight) in enumerate(zip(classes, weights)):
        print(f"  {cls:>8}: {weight:.3f}")
    
    return class_weight_dict, classes

def train_improved_model(data_dir, output_dir, epochs=50):
    """Train improved model with balanced dataset"""
    
    print("Starting improved model training...")
    
    # Setup paths
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute class weights
    class_weights, class_names = compute_class_weights(data_dir)
    
    # Load balanced dataset
    print("\nLoading balanced dataset...")
    train_ds, val_ds = load_balanced_dataset(data_dir)
    
    # Create improved model
    print("\nCreating improved model...")
    num_classes = len(class_names)
    model = create_improved_model(num_classes=num_classes)
    
    # Compile with better optimizer and learning rate schedule
    initial_lr = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.h5",
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    model.save(output_dir / "cell_cnn_improved.h5")
    
    # Save class names
    with open(output_dir / "classes.json", "w") as f:
        json.dump(class_names, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir / 'cell_cnn_improved.h5'}")
    print(f"Classes saved to: {output_dir / 'classes.json'}")
    
    # Print final metrics
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    
    return model, history

if __name__ == "__main__":
    import sys
    
    # Check if balanced dataset exists
    ROOT = Path(".").resolve()
    BALANCED_DIR = ROOT / "data/balanced/cells"
    
    if not BALANCED_DIR.exists():
        print("ERROR: Balanced dataset not found!")
        print(f"Expected: {BALANCED_DIR}")
        print("\nRun the dataset balancing script first:")
        print("python scripts/fix_dataset_balance.py")
        sys.exit(1)
    
    # Check if we have enough classes
    class_dirs = [d for d in BALANCED_DIR.iterdir() if d.is_dir()]
    if len(class_dirs) < 13:
        print(f"ERROR: Only found {len(class_dirs)} classes, expected 13")
        print("Run the dataset balancing script first.")
        sys.exit(1)
    
    OUTPUT_DIR = ROOT / "models"
    
    print(f"Training improved model...")
    print(f"Data: {BALANCED_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    model, history = train_improved_model(
        data_dir=BALANCED_DIR,
        output_dir=OUTPUT_DIR,
        epochs=50
    )
    
    print("\nImproved model training complete!")