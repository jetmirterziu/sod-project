import os
import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from data_loader import get_data_loaders
from sod_model import SODModel

# --- HYPERPARAMETERS & CONFIGURATION ---
LEARNING_RATE = 1e-3
EPOCHS = 25 
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "results"
DATA_DIR = "data/raw" 

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- CUSTOM LOSS FUNCTION ---
# Requirement: Binary Cross-Entropy + 0.5 * (1 - IoU)
def sod_loss(y_true, y_pred):
    # 1. Binary Cross Entropy (BCE)
    # Measures the difference between probability distributions (mask vs. prediction)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    # 2. Intersection over Union (IoU)
    smooth = 1e-6 # For numerical stability (prevents division by zero)
    
    # Flatten the image/mask tensors for easier metric calculation
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    # Combined Loss: We minimize (1 - IoU) and scale it by 0.5
    loss = bce + 0.5 * (1.0 - iou)
    return loss

# --- CUSTOM METRIC ---
def calculate_iou(y_true, y_pred, threshold=0.5):
    """Calculates IoU, used for tracking performance (not for backprop)."""
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)
    
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_bin_f = tf.reshape(y_pred_bin, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_bin_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_bin_f) - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# --- TRAINING SCRIPT ---
def train():
    # 1. Load Data
    print("Loading dataset...")
    try:
        train_ds, val_ds, _ = get_data_loaders(DATA_DIR)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Initialize Model, Optimizer, and Checkpoint Manager
    model = SODModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Checkpoint Manager (BONUS TASK: Save and Resume)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

    # Load the latest checkpoint if it exists
    start_epoch = 0
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        start_epoch = int(ckpt.step)
        print(f"Restored from epoch {start_epoch}. Resuming from epoch {start_epoch + 1}.")
    else:
        print("Starting training from scratch.")
        
    # --- TRAINING STEP FUNCTION (Manual Backpropagation) ---
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            # Forward Pass
            logits = model(x_batch, training=True)
            loss = sod_loss(y_batch, logits)
        
        # Backward Pass: Calculate and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calculate metrics for logging
        iou = calculate_iou(y_batch, logits)
        return loss, iou

    # --- VALIDATION STEP FUNCTION ---
    @tf.function
    def val_step(x_batch, y_batch):
        # Forward Pass (training=False ensures no updates to Batch Norm stats or Dropout)
        val_logits = model(x_batch, training=False)
        v_loss = sod_loss(y_batch, val_logits)
        v_iou = calculate_iou(y_batch, val_logits)
        return v_loss, v_iou

    # 3. Training Loop Execution
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5 # Early stopping patience

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        start_time = time.time()

        # Reset epoch metrics
        train_loss_avg = tf.keras.metrics.Mean()
        train_iou_avg = tf.keras.metrics.Mean()

        # Train loop
        for x_batch, y_batch in tqdm(train_ds, desc="Training"):
            loss, iou = train_step(x_batch, y_batch)
            train_loss_avg.update_state(loss)
            train_iou_avg.update_state(iou)

        # Reset validation metrics
        val_loss_avg = tf.keras.metrics.Mean()
        val_iou_avg = tf.keras.metrics.Mean()
        
        # Validation loop (no gradient calculation)
        for x_val, y_val in val_ds:
            v_loss, v_iou = val_step(x_val, y_val)
            val_loss_avg.update_state(v_loss)
            val_iou_avg.update_state(v_iou)

        # --- LOGGING & CHECKPOINTING ---
        current_train_loss = train_loss_avg.result()
        current_val_loss = val_loss_avg.result()

        print(f"Train Loss: {current_train_loss:.4f}, Train IoU: {train_iou_avg.result():.4f}")
        print(f"Val Loss:   {current_val_loss:.4f}, Val IoU:   {val_iou_avg.result():.4f}")
        print(f"Time: {time.time() - start_time:.2f}s")

        # Save Checkpoint (Bonus Requirement)
        ckpt.step.assign_add(1)
        manager.save()
        
        # Early Stopping
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
            # Save the BEST model weights (for use in evaluate.py)
            model.save_weights(os.path.join(LOG_DIR, "best_model.weights.h5"))
            print("New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered. Model stopped.")
                break

if __name__ == "__main__":
    train()