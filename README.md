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
#### Metrics (Test Set)
| Metric	| Score	|
| :--- | :--- |
|Mean IoU|	61.5%	|
|F1-Score|	73.8%	|
|Precision|	76.5%	|
|Recall|	77.4%	|

#### Experiments

### Visual Demonstration
#### Overlay Visualization
The system outputs a probability mask which is overlaid (red heatmap) onto the original image.
 <br>```See results folder.```


 ### Author
 *Jetmir Terziu*
 <br>**Xponian Cohort IV - AI Stream**
 <br>Mentor: *Berat Ujkani*