# Binary ECG Signal Classification with Bidirectional GRU

## Project Summary

This project focuses on **binary classification of ECG signals** into two classes:

* **Normal**
* **Abnormal**

The model is trained on the **PTB Diagnostic ECG Database (PTBDB)** and uses a **Bidirectional GRU (BiGRU)** architecture to learn temporal patterns from ECG signal sequences.

The project includes:

* ECG signal preprocessing and dataset preparation
* Train/test splitting with stratification
* Class balancing using upsampling
* Bidirectional GRU-based sequence classification
* Binary classification with sigmoid output
* Model evaluation using classification metrics
* Confusion matrix visualization
* Saving the trained model for later inference

---

## Dataset

The project uses the **PTBDB dataset**, consisting of two classes:

| Class    | Label |
| -------- | ----: |
| Normal   |     0 |
| Abnormal |     1 |

The original data is loaded from:

```text
ptbdb_normal.csv
ptbdb_abnormal.csv
```

Each sample contains **187 signal values**, followed by its class label.

### Dataset Preparation

The data is first divided into training and test sets using an **80/20 split** with stratification.

The training set is then balanced using **upsampling** of the Normal class.

After balancing:

* **Training samples:** 16,808
* **Test samples:** 2,911
* **Input shape:** `(187, 1)`

The test set remains separate from the balancing process.

## Model Architecture

The classifier is based on a **Bidirectional GRU (BiGRU)** architecture.

### Architecture

```text
Input ECG Signal
      │
      ▼
Masking
      │
      ▼
Bidirectional GRU (128 units)
      │
      ▼
Bidirectional GRU (64 units)
      │
      ▼
Dense (128, ReLU)
      │
      ▼
Dropout (0.2)
      │
      ▼
Dense (64, ReLU)
      │
      ▼
Dense (1, Sigmoid)
      │
      ▼
Normal / Abnormal
```

### Model Configuration

* Input shape: `187 × 1`
* Masking value: `0.0`
* Bidirectional GRU: 128 units
* Bidirectional GRU: 64 units
* Dense: 128 units, ReLU
* Dropout: 0.2
* Dense: 64 units, ReLU
* Output: 1 neuron with sigmoid activation
* Total parameters: **249,089**
* Loss: Binary Cross-Entropy
* Optimizer: Adam
* Learning rate: `0.0005`

The complete model architecture and parameter count are defined in the notebook.

## Training

The model was trained with:

| Parameter               |                Value |
| ----------------------- | -------------------: |
| Maximum Epochs          |                   50 |
| Batch Size              |                   64 |
| Validation Split        |                  15% |
| Learning Rate           |               0.0005 |
| Loss Function           | Binary Cross-Entropy |
| Optimizer               |                 Adam |
| Early Stopping Patience |                   12 |
| LR Reduction Factor     |                  0.2 |

Two callbacks were used during training:

* **EarlyStopping** based on validation loss
* **ReduceLROnPlateau** to reduce the learning rate when validation loss stopped improving

The learning rate was progressively reduced during training.

---

## Results

The final model was evaluated on the held-out test set containing **2,911 ECG samples**.

### Classification Performance

| Class        | Precision | Recall | F1-Score |   Support |
| ------------ | --------: | -----: | -------: | --------: |
| Normal       |      0.92 |   0.97 |     0.94 |       809 |
| Abnormal     |      0.99 |   0.97 |     0.98 |     2,102 |
| **Accuracy** |           |        | **0.97** | **2,911** |

Additional metrics:

* **Macro F1:** 0.96
* **Weighted F1:** 0.97
* **Test Accuracy:** **97%**

The notebook also generates a confusion matrix to visualize the classification results.

---

## Trained Model

The trained model is saved as:

```text
models/ECG_Binary_Classifier_97.h5
```

### Loading the Model

```python
from tensorflow.keras.models import load_model

model = load_model("models/ECG_Binary_Classifier_97.h5")
```

The saved model can then be used for inference on ECG samples with the same input format used during training:

```text
(187, 1)
```

---

## Prediction

The model produces a probability using the sigmoid output.

A threshold of `0.5` is used to convert the probability into a binary prediction:

```python
prediction = (model.predict(X_test) > 0.5).astype("int32")
```

The output classes are:

```text
0 → Normal
1 → Abnormal
```
---

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Key Concepts

This project demonstrates the application of deep learning to sequential biomedical signals, with a focus on:

* ECG signal classification
* Time-series modeling
* Recurrent Neural Networks
* Bidirectional GRU
* Binary classification
* Class imbalance handling
* Model evaluation
* Confusion matrix analysis

---

## Course Information

**Deep Learning Course**
University of Isfahan

This project was developed as part of the coursework for studying deep learning methods and their application to sequential signal classification.
