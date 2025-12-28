import tensorflow as tf
import h5py
import numpy as np
import os

def diagnose_model(model_path):
    print(f"🔍 Diagnosing model: {model_path}")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    
    try:
        # Method 1: Try to load with Keras
        print("\n1️⃣ Attempting to load with tf.keras.models.load_model...")
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Successfully loaded with Keras!")
            print(f"   Model type: {type(model)}")
            print(f"   Model name: {model.name}")
            print(f"   Number of layers: {len(model.layers)}")
            
            # Show layer summary
            print("\n   Layer summary:")
            model.summary()
            
            # Check if it's a Functional or Sequential model
            if isinstance(model, tf.keras.Sequential):
                print("   📋 Model is Sequential")
            elif isinstance(model, tf.keras.Model):
                print("   🔧 Model is Functional")
                
            return model
            
        except Exception as e:
            print(f"❌ Keras loading failed: {e}")
    
    except Exception as e:
        print(f"❌ Error during Keras load: {e}")
    
    try:
        # Method 2: Inspect H5 file structure
        print("\n2️⃣ Inspecting H5 file structure...")
        with h5py.File(model_path, 'r') as f:
            print("   H5 file keys:")
            for key in f.keys():
                print(f"     - {key}")
                if hasattr(f[key], 'keys'):
                    for subkey in f[key].keys():
                        print(f"       └── {subkey}")
                        
            # Check for model config
            if 'model_config' in f.keys():
                print("   📄 Model config found")
                import json
                config = json.loads(f.attrs['model_config'])
                if 'class_name' in config:
                    print(f"   🏷️ Model class: {config['class_name']}")
                    
            if 'model_weights' in f.keys():
                print("   ⚖️ Model weights found")
                
    except Exception as e:
        print(f"❌ H5 inspection failed: {e}")
    
    try:
        # Method 3: Try loading weights only
        print("\n3️⃣ Attempting to load architecture and weights separately...")
        
        # Try different architectures
        architectures = [
            ("Sequential VGG16", create_sequential_vgg16()),
            ("Functional VGG16", create_functional_vgg16()),
            ("Custom CNN", create_custom_cnn())
        ]
        
        for arch_name, model in architectures:
            try:
                print(f"   Trying {arch_name}...")
                model.load_weights(model_path)
                print(f"   ✅ Weights loaded successfully for {arch_name}!")
                print(f"   Model summary for {arch_name}:")
                model.summary()
                return model
            except Exception as e:
                print(f"   ❌ {arch_name} failed: {str(e)[:100]}...")
                
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")

def create_sequential_vgg16():
    """Create VGG16 as Sequential model"""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
    
    vgg_base = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    vgg_base.trainable = False
    
    model = tf.keras.Sequential([
        vgg_base,
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    return model

def create_functional_vgg16():
    """Create VGG16 as Functional model"""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
    from tensorflow.keras import Model
    
    vgg_base = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    vgg_base.trainable = False
    
    x = vgg_base.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=vgg_base.input, outputs=predictions)
    return model

def create_custom_cnn():
    """Create Custom CNN model"""
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    
    model = tf.keras.Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Test all model files
    model_files = ['vgg16_model.h5', 'custom_cnn_model.h5', 'fine_tuned_vgg16.h5']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            diagnose_model(model_file)
            print("\n" + "=" * 80 + "\n")
        else:
            print(f"⚠️ {model_file} not found\n")