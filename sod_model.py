import tensorflow as tf
from keras import layers, models, Input

def SODModel(input_shape=(128, 128, 3)):
    """
    Improved Encoder-Decoder CNN for Salient Object Detection.
    Modifications for Experiment:
    1. Added BatchNormalization (Stabilize training).
    2. Added Dropout (Reduce overfitting).
    """
    inputs = Input(shape=input_shape)

    # --- ENCODER (Downsampling) ---
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs) 
    x = layers.BatchNormalization()(x) # IMPROVEMENT 1: Batch Norm
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x) 

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x) # IMPROVEMENT 1
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x) 
    
    x = layers.Dropout(0.3)(x) # IMPROVEMENT 2: Dropout

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x) # IMPROVEMENT 1
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x) 
    
    # Bottleneck 
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Dropout(0.4)(x) # IMPROVEMENT 2

    # --- DECODER (Upsampling) ---
    # Block 3 Upsample
    x = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x) # IMPROVEMENT 1
    x = layers.ReLU()(x)
    
    # Block 2 Upsample
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x) # IMPROVEMENT 1
    x = layers.ReLU()(x)
    
    # Block 1 Upsample
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.ReLU()(x)

    # --- OUTPUT ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Improved_SOD_Model")
    
    return model

if __name__ == "__main__":
    model = SODModel()
    model.summary()