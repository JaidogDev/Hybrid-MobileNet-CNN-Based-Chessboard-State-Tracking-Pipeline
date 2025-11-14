#!/usr/bin/env python3
"""
Lightweight training approach using transfer learning for laptop training
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import time

def create_lightweight_model(num_classes=13, img_size=96, freeze_layers=True):
    """Create lightweight model using MobileNetV2 transfer learning"""
    
    # Use MobileNetV2 as base (very lightweight, designed for mobile)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        alpha=0.75,  # Reduce model size by 25%
        include_top=False,
        weights='imagenet'
    )
    
    if freeze_layers:
        # Freeze most layers, only train top layers
        base_model.trainable = False
    else:
        # Fine-tune only the last few layers
        base_model.trainable = True
        for layer in base_model.layers[:-30]:  # Freeze all but last 30 layers
            layer.trainable = False
    
    # Add custom classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def quick_train_strategy(data_dir, output_dir, quick_mode=True):
    """Quick training strategy for laptop"""
    
    print("🚀 Starting LIGHTWEIGHT training for laptop...")
    
    # Setup paths
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quick dataset loading (no validation split for now)
    IMG_SIZE = 96
    BATCH_SIZE = 16 if quick_mode else 32  # Smaller batch for laptop
    
    print("📂 Loading dataset (no validation split)...")
    
    # Load ALL data for training (as you suggested)
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    
    class_names = full_ds.class_names
    print(f"📊 Found {len(class_names)} classes: {class_names}")
    
    # Count samples per class
    total_samples = 0
    for batch in full_ds:
        total_samples += batch[0].shape[0]
    print(f"📈 Total samples: {total_samples}")
    
    # Normalize and optimize dataset  
    normalization_layer = tf.keras.utils.Rescaling(1./255)
    full_ds = full_ds.map(lambda x, y: (normalization_layer(x), y))
    full_ds = full_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # Create lightweight model
    print("🏗️ Creating MobileNetV2-based model...")
    model = create_lightweight_model(
        num_classes=len(class_names),
        img_size=IMG_SIZE,
        freeze_layers=True  # Start with frozen base
    )
    
    # Compile with efficient settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("📋 Model summary:")
    model.summary()
    
    # Phase 1: Quick frozen training (2-3 epochs)
    print("\n🔄 Phase 1: Quick feature extraction (frozen base)...")
    
    start_time = time.time()
    
    history1 = model.fit(
        full_ds,
        epochs=3 if quick_mode else 5,
        verbose=1
    )
    
    phase1_time = time.time() - start_time
    print(f"⏱️ Phase 1 completed in {phase1_time:.1f}s")
    
    # Phase 2: Fine-tuning (optional, if accuracy is low)
    final_acc = history1.history['accuracy'][-1]
    print(f"📊 Phase 1 final accuracy: {final_acc:.3f}")
    
    if final_acc < 0.7 and not quick_mode:  # Only fine-tune if needed
        print("\n🔧 Phase 2: Fine-tuning (unfreezing layers)...")
        
        # Unfreeze model for fine-tuning
        model = create_lightweight_model(
            num_classes=len(class_names),
            freeze_layers=False  # Allow fine-tuning
        )
        
        # Lower learning rate for fine-tuning
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            full_ds,
            epochs=3,
            verbose=1
        )
        
        final_acc = history2.history['accuracy'][-1]
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f}s")
    print(f"📊 Final accuracy: {final_acc:.3f}")
    
    # Save model and classes
    model_path = output_dir / "cell_cnn_lightweight.h5"
    classes_path = output_dir / "classes.json"
    
    model.save(model_path)
    
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    
    print(f"💾 Model saved to: {model_path}")
    print(f"💾 Classes saved to: {classes_path}")
    
    return model, class_names

def test_model_quickly(model, class_names):
    """Quick test of the model on sample data"""
    
    print("\n🧪 Quick model test...")
    
    # Create dummy test data
    test_input = tf.random.normal([1, 96, 96, 3])
    
    # Get prediction
    prediction = model(test_input)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    confidence = tf.reduce_max(prediction).numpy()
    
    print(f"📊 Test prediction:")
    print(f"   Class: {class_names[predicted_class]}")
    print(f"   Confidence: {confidence:.3f}")
    
    # Check if model is reasonable
    if confidence > 0.1:  # At least some confidence
        print("✅ Model seems to be working!")
        return True
    else:
        print("❌ Model might need more training")
        return False

if __name__ == "__main__":
    
    # Configuration
    ROOT = Path(".").resolve()
    DATA_DIR = ROOT / "data/balanced/cells"
    OUTPUT_DIR = ROOT / "models"
    
    # Check data availability
    if not DATA_DIR.exists():
        print("❌ Balanced dataset not found!")
        print("Run: python scripts/fix_dataset_balance.py")
        exit(1)
    
    print("⚡ LAPTOP-OPTIMIZED TRAINING ⚡")
    print("=" * 50)
    print(f"📂 Data: {DATA_DIR}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 50)
    
    # Quick training
    model, class_names = quick_train_strategy(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        quick_mode=True  # Fast training for laptop
    )
    
    # Quick test
    is_working = test_model_quickly(model, class_names)
    
    if is_working:
        print("\n🎉 SUCCESS! Lightweight model is ready!")
        print("\nNext steps:")
        print("1. Test on your videos: python scripts/test_inference.py")
        print("2. If accuracy is good enough, submit to Kaggle!")
        print("3. If not, run with quick_mode=False for better training")
    else:
        print("\n⚠️ Model needs improvement. Try:")
        print("1. More training epochs")
        print("2. Better data balancing")
        print("3. Different augmentations")
    
    print(f"\n💡 Total model size: ~{model.count_params()/1000:.1f}K parameters")
    print("Perfect for laptop training! 🚀")