import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from data_loader import get_data_loaders
from sod_model import SODModel

# --- CONFIGURATION ---
DATA_DIR = "data/raw"
# Matches the file saved by train.py
WEIGHTS_PATH = "results/best_model.weights.h5" 
RESULTS_DIR = "results"
BATCH_SIZE = 16

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculates IoU, Precision, Recall, and F1-Score for a single batch.
    """
    # 1. Force Ground Truth to be Binary (0 or 1)
    # This fixes the "continuous" error if resizing created values like 0.99 or 0.01
    y_true_bin = (y_true > 0.5).astype(np.int32)
    
    # 2. Force Prediction to be Binary (0 or 1)
    y_pred_bin = (y_pred > threshold).astype(np.int32)

    # 3. Flatten for Scikit-Learn
    y_true_f = y_true_bin.flatten()
    y_pred_f = y_pred_bin.flatten()
    
    # Precision, Recall, F1
    precision = precision_score(y_true_f, y_pred_f, zero_division=0)
    recall = recall_score(y_true_f, y_pred_f, zero_division=0)
    f1 = f1_score(y_true_f, y_pred_f, zero_division=0)
    
    # IoU (Intersection over Union)
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou, precision, recall, f1

def visualize_results(model, test_ds, num_samples=5):
    """
    Generates a visualization grid: Input | Ground Truth | Prediction | Overlay
    """
    print("Generating visualizations...")
    
    # Take one batch from the test set
    for images, masks in test_ds.take(1):
        preds = model.predict(images)
        
        # Create a figure with rows = num_samples
        plt.figure(figsize=(12, num_samples * 3))
        
        for i in range(min(num_samples, len(images))):
            img = images[i].numpy()
            gt = masks[i].numpy().squeeze()
            pred = preds[i].squeeze()
            
            # --- OVERLAY LOGIC ---
            # Create a red heatmap for the prediction
            heatmap = np.zeros_like(img)
            heatmap[:, :, 0] = pred  # Red channel gets the prediction
            
            # Blend: 70% Original Image + 30% Red Heatmap
            overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
            
            # 1. Input Image
            plt.subplot(num_samples, 4, i * 4 + 1)
            plt.imshow(img)
            plt.title("Input")
            plt.axis('off')

            # 2. Ground Truth (Actual Mask)
            plt.subplot(num_samples, 4, i * 4 + 2)
            plt.imshow(gt, cmap='gray')
            plt.title("Ground Truth")
            plt.axis('off')

            # 3. Prediction (Model Output)
            plt.subplot(num_samples, 4, i * 4 + 3)
            plt.imshow(pred, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
            
            # 4. Overlay (Visual check)
            plt.subplot(num_samples, 4, i * 4 + 4)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "test_predictions.png")
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close()

def evaluate():
    # 1. Load Test Data
    print("Loading test data...")
    try:
        # We only need test_ds here
        _, _, test_ds = get_data_loaders(DATA_DIR)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Load Model Architecture
    print("Loading model...")
    model = SODModel()
    
    # Run a dummy pass to initialize the model variables so we can load weights
    dummy_input = tf.zeros((1, 128, 128, 3))
    model(dummy_input)
    
    # 3. Load Trained Weights
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
        print(f"Loaded weights from {WEIGHTS_PATH}")
    else:
        print(f"ERROR: Weights not found at {WEIGHTS_PATH}. Run train.py first!")
        return

    # 4. Quantitative Evaluation (The Numbers)
    print("Calculating metrics on Test Set...")
    iou_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    # Loop through the Test Set
    for x_batch, y_batch in tqdm(test_ds, desc="Evaluating"):
        preds = model.predict(x_batch, verbose=0)
        
        # Calculate metrics for each image in the batch
        for i in range(len(x_batch)):
            iou, prec, rec, f1 = calculate_metrics(y_batch[i].numpy(), preds[i])
            iou_scores.append(iou)
            prec_scores.append(prec)
            rec_scores.append(rec)
            f1_scores.append(f1)

    # Print Final Results
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Mean IoU:       {np.mean(iou_scores):.4f}")
    print(f"Mean Precision: {np.mean(prec_scores):.4f}")
    print(f"Mean Recall:    {np.mean(rec_scores):.4f}")
    print(f"Mean F1-Score:  {np.mean(f1_scores):.4f}")
    print("="*30)
    
    # 5. Qualitative Evaluation (The Visuals)
    visualize_results(model, test_ds)

if __name__ == "__main__":
    evaluate()