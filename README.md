# Cassava Leaf Disease Classification

Multi-class image classification of cassava leaf diseases using transfer learning with ResNet-50. Built for the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) competition.

Developed as the final project for CS 570 – Deep Learning (Fall 2021) at the University of Idaho, under Dr. Jamil.

---

## Problem

Cassava is a key food crop across sub-Saharan Africa, but viral diseases can destroy entire harvests. This project trains a convolutional neural network to classify cassava leaf images into 5 categories:

| Label | Disease |
|---|---|
| 0 | Cassava Bacterial Blight (CBB) |
| 1 | Cassava Brown Streak Disease (CBSD) |
| 2 | Cassava Green Mottle (CGM) |
| 3 | Cassava Mosaic Disease (CMD) |
| 4 | Healthy |

---

## Architecture

ResNet-50 pretrained on ImageNet, fine-tuned for 5-class classification:

```
Input (512×512×3)
  → Data Augmentation (random flip, rotation, crop)
  → ResNet-50 backbone (ImageNet weights, trainable)
  → GlobalMaxPooling2D
  → BatchNormalization
  → Dense(512, ReLU) + L1L2 regularization
  → Dropout(0.3)
  → BatchNormalization
  → Dense(256, ReLU) + L1L2 regularization
  → Dropout(0.3)
  → Dense(5, Softmax)
```

> Architecture diagram: ResNet-50 backbone with 5 blocks (64 → 128 → 256 → 512 filters), residual skip connections, followed by GlobalMaxPooling and fully connected head.

---

## Results

Four submission versions were generated during development:

| Version | Description |
|---|---|
| v1 | Baseline ResNet-50, 10 epochs, lr=1e-3 |
| v2 | Same model, path fix for test image IDs |
| v3 | Tuned hyperparameters |
| v4 | Final model: augmentation + regularization + lr=1e-4 |

All prediction files are in the [`results/`](results/) folder.

---

## Project Structure

```
.
├── notebooks/
│   ├── DL_final_project_final.ipynb   # Final training notebook (run on Google Colab)
│   └── Data_Loader.ipynb              # Data loading and preprocessing experiments
├── results/
│   ├── sample_submission.csv          # Competition format reference
│   ├── submission_v1.csv              # Baseline predictions
│   ├── submission_v2.csv
│   ├── submission_v3.csv
│   └── submission_v4.csv              # Final submission
├── .gitignore
└── README.md
```

---

## Running the Notebook

The notebooks are designed to run on **Google Colab** with a GPU runtime and the dataset stored in Google Drive.

### Setup

1. Go to [Kaggle – Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification/data) and download the dataset.
2. Upload the zip to your Google Drive under:
   ```
   MyDrive/Colab Notebooks/cassava-leaf-disease-classification/
   ```
3. Open `notebooks/DL_final_project_final.ipynb` in [Google Colab](https://colab.research.google.com/).
4. Set the runtime to **GPU** (Runtime → Change runtime type → GPU).
5. Run all cells.

### Training configuration

| Parameter | Value |
|---|---|
| Image size | 512 × 512 |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam (lr = 1e-4) |
| Loss | Categorical cross-entropy |
| Validation split | 20% |

---

## Dependencies

All dependencies are pre-installed in Google Colab. For local use:

```bash
pip install tensorflow keras pandas numpy matplotlib
```

| Library | Version used |
|---|---|
| TensorFlow | 2.x |
| Keras | bundled with TF |
| NumPy | any |
| Pandas | any |
| Matplotlib | any |

---

## Acknowledgements

- Dataset: [Kaggle – Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)
- Backbone: [ResNet-50](https://arxiv.org/abs/1512.03385) pretrained on ImageNet via `tf.keras.applications`
- Course: CS 570 – Deep Learning, Fall 2021, University of Idaho
