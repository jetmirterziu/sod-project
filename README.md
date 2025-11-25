# Salient Object Detection (SOD) - End-to-End ML Pipeline

An end-to-end Deep Learning project that detects and segments the most visually important object in an image. Designed and implemented **strictly from scratch** without pre-trained backbones, featuring a custom CNN architecture and training loop.

---

## Project Overview

This project implements a **Salient Object Detection (SOD)** system using a custom **Encoder-Decoder Convolutional Neural Network**.

### Key Features
* **Architecture from Scratch:** A custom 4-layer Encoder-Decoder network.
* **Custom Training Loop:** Implemented using `tf.GradientTape` for manual backpropagation.
* **Hybrid Loss Function:** Combines **Binary Cross-Entropy** (pixel accuracy) with **IoU Loss** (shape overlap) for sharper segmentation.
* **Pipeline:** Includes `tf.data` prefetching, augmentation (Flip/Crop), and checkpoint recovery.
* **Interactive Demo:** A deployed **Streamlit** web app for real-time inference.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `data_loader.py` | ETL pipeline. Handles image resizing, normalization, and augmentation (Random Crop/Flip). |
| `sod_model.py` | Defines the custom Encoder-Decoder CNN architecture. |
| `train.py` | The training engine. Contains the custom training loop, hybrid loss function, and checkpoint logic. |
| `evaluate.py` | Calculates quantitative metrics (IoU, F1) and generates visual overlays. |
| `app.py` | Streamlit web application for the live user demo. |
| `requirements.txt` | List of dependencies (TensorFlow, OpenCV, Streamlit, etc.). |

---

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/jetmirterziu/sod-project.git
    cd sod-project
    ```

2.  **Create Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\Activate

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Dataset**
    * Download the **MSRA10K**.
    * Organize files into `data/raw/images/` and `data/raw/masks/`.

---

## Usage

### 1. Training
Train the model from scratch. This script automatically saves checkpoints to `checkpoints/` and the best model to `results/`.
```bash
python train.py
```
Configuration: 25 Epochs, Batch Size 16, Adam Optimizer (1e-3).

### 2. Evaluation
Calculate Mean IoU, Precision, Recall, and F1-Score on the Test Set.
```bash
python evaluate.py
```

### 3. Live Demo
Launch the web interface to test the model on your own images.
```bash
streamlit run app.py
```

### 4. Results & Experiments

### Quantitative Metrics (Test Set - Improved Model)
| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Mean IoU** | **61.8%** | Improvement over baseline (61.5%). |
| **F1-Score** | **74.1%** | Balanced performance with a focus on structural accuracy. |
| **Precision** | **78.1%** | Significant increase (+1.6%) indicates reduced background noise. |
| **Recall** | **76.9%** | Stable object coverage. |

### Experiments Comparison
To improve beyond the baseline, two architectural changes were implemented (BatchNormalization + Dropout).

| Experiment | Modifications | Mean IoU | Min Val Loss | Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Standard Conv2D + ReLU | **61.5%** | **0.5393** | Good start, but higher loss indicates less stability. |
| **Improved** | **1. BatchNormalization**<br>**2. Dropout (0.3/0.4)** | **61.8%** | **0.5205** | **Lower Loss:** Model is more confident.<br>**Higher Precision:** Cleaner object boundaries. |

### Visual Demonstration
#### Overlay Visualization
The system outputs a probability mask which is overlaid (red heatmap) onto the original image.
 <br>```See results folder.```


 ### Author
 *Jetmir Terziu*
 <br>**Xponian Cohort IV - AI Stream**
 <br>Mentor: *Berat Ujkani*