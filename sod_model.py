import tensorflow as tf
from keras import layers, models, Input

def SODModel(input_shape=(128, 128, 3)):
    """
    Creates a simple Encoder-Decoder CNN for Salient Object Detection,
    built from scratch using Keras layers.
    
    Args:
        input_shape: Tuple representing input image resolution (H, W, C).
        
    Returns:
        model: A tf.keras.Model.
    """
    inputs = Input(shape=input_shape)

    # --- ENCODER (Feature Extraction & Downsampling) ---
    # Block 1: 128x128 -> 64x64
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x) 

    # Block 2: 64x64 -> 32x32
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x) 

    # Block 3: 32x32 -> 16x16
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x) 
    
    # Bottleneck (Latent Space)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

    # --- DECODER (Feature Reconstruction & Upsampling) ---
    # Block 3 Upsample: 16x16 -> 32x32
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.ReLU()(x)
    
    # Block 2 Upsample: 32x32 -> 64x64
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.ReLU()(x)
    
    # Block 1 Upsample: 64x64 -> 128x128
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.ReLU()(x)

    # --- OUTPUT LAYER ---
    # 1 Channel output, Sigmoid activation for binary prediction (0 to 1)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Simple_SOD_Model")
    
    return model

# Quick check to verify architecture
if __name__ == "__main__":
    model = SODModel() 
    model.summary()
    print("\nModel created successfully!")