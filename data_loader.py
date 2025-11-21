import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob

# --- CONFIGURATION ---
IMG_HEIGHT = 128
IMG_WIDTH = 128
# We resize training images slightly larger to allow for random cropping
IMG_HEIGHT_AUG = 144 
IMG_WIDTH_AUG = 144
BATCH_SIZE = 16
RANDOM_SEED = 42

def load_files(image_path, mask_path):
    """
    Loads raw bytes from files.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    
    return img, mask

def preprocess_train(image_path, mask_path):
    """
    Preprocessing for TRAINING set (includes resizing for crop).
    """
    img, mask = load_files(image_path, mask_path)
    
    # Resize to slightly larger dimension to allow random crop later
    img = tf.image.resize(img, [IMG_HEIGHT_AUG, IMG_WIDTH_AUG])
    mask = tf.image.resize(mask, [IMG_HEIGHT_AUG, IMG_WIDTH_AUG], method='nearest')
    
    # Normalize to 0-1
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    
    return img, mask

def preprocess_test(image_path, mask_path):
    """
    Preprocessing for VALIDATION/TEST set (exact target size).
    """
    img, mask = load_files(image_path, mask_path)
    
    # Resize directly to target size
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    
    # Normalize to 0-1
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    
    return img, mask

def augment(image, mask):
    """
    Applies random augmentations to the training pairs.
    Includes: Random Crop, Horizontal Flip, Brightness.
    """
    # 1. Random Crop (from 144x144 down to 128x128)
    # Stack them to crop exactly the same region
    combined = tf.concat([image, mask], axis=-1) 
    combined = tf.image.random_crop(combined, size=[IMG_HEIGHT, IMG_WIDTH, 4])
    
    # Unstack
    image = combined[:, :, :3]
    mask = combined[:, :, 3:]

    # 2. Horizontal Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # 3. Brightness (Image only)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

def get_data_loaders(dataset_dir):
    """
    Creates Train, Val, and Test loaders.
    Expects 'images' and 'masks' folders inside dataset_dir.
    """
    # 1. Get Paths
    img_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    
    # Sort to ensure alignment
    img_paths = sorted(glob(os.path.join(img_dir, "*")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*")))

    # Basic Check
    if len(img_paths) == 0 or len(mask_paths) == 0:
        raise FileNotFoundError(f"No images found in {dataset_dir}. Check 'images' and 'masks' folders.")
    
    # 2. Split Data (70% Train, 15% Val, 15% Test)
    train_img, temp_img, train_mask, temp_mask = train_test_split(
        img_paths, mask_paths, test_size=0.3, random_state=RANDOM_SEED
    )
    val_img, test_img, val_mask, test_mask = train_test_split(
        temp_img, temp_mask, test_size=0.5, random_state=RANDOM_SEED
    )

    print(f"Dataset Loaded: {len(train_img)} Train, {len(val_img)} Val, {len(test_img)} Test")

    # 3. Create TensorFlow Datasets
    
    # Train Dataset (Uses Augmentation)
    train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_mask))
    train_ds = train_ds.map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Val Dataset (No Augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_mask))
    val_ds = val_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Test Dataset (No Augmentation)
    test_ds = tf.data.Dataset.from_tensor_slices((test_img, test_mask))
    test_ds = test_ds.map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Path to where your 'images' and 'masks' folders are
    DATA_PATH = "data/raw" 
    
    if os.path.exists(DATA_PATH) and os.path.exists(os.path.join(DATA_PATH, "images")):
        try:
            train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH)
            
            # Fetch one batch to verify
            for img_batch, mask_batch in train_loader.take(1):
                print(f"Train Batch Image Shape: {img_batch.shape}") # Should be (16, 128, 128, 3)
                print(f"Train Batch Mask Shape: {mask_batch.shape}") # Should be (16, 128, 128, 1)
                print("Data Pipeline is working correctly!")
        except Exception as e:
            print(f"Error during loading: {e}")
    else:
        print(f"Data path '{DATA_PATH}' not found!")