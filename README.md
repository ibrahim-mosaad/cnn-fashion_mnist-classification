# Fashion-MNIST Image Classification using CNN

This repository features an end-to-end Deep Learning pipeline to classify clothing items from the **Fashion-MNIST** dataset. Built using **TensorFlow** and **Keras**, the project demonstrates the implementation of a Convolutional Neural Network (CNN) following industry best practices for data handling, visualization, and model persistence.

##  Overview

The Fashion-MNIST dataset is a collection of 70,000 grayscale images in 10 categories (T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots). This notebook achieves high classification accuracy by leveraging convolutional layers for spatial feature extraction.

##  Tech Stack

* **Framework:** TensorFlow 2.19.0
* **Data Loading:** TensorFlow Datasets (TFDS)
* **Deep Learning API:** Keras (Sequential API)
* **Visualization:** Matplotlib, NumPy
* **Environment:** Jupyter Notebook / Kaggle (GPU Accelerated)

##  Project Structure

1. **Dependencies:** Loading core libraries and configuring logging for a clean output.
2. **Data Acquisition:** Using `tfds` for a reproducible train/test split.
3. **Preprocessing:** Feature scaling (normalization) and adding channel dimensions for CNN compatibility.
4. **EDA:** Visualizing single and batch samples to understand class labels.
5. **Model Architecture:** * `Conv2D` + `MaxPooling2D` blocks for feature learning.
* `Flatten` + `Dense` (ReLU) layers for classification.
* `Softmax` output for 10-class probability distribution.


6. **Training Pipeline:** Utilizing `shuffle`, `batch`, and `repeat` for high-performance input pipelines.
7. **Evaluation:** Validation on unseen data and performance metrics reporting.
8. **Inference Visualization:** Comparing predicted labels vs. true labels with visual confidence checks.

##  Model Architecture

```text
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320       
 max_pooling2d (MaxPooling)  (None, 14, 14, 32)        0         
 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496     
 max_pooling2d_1 (MaxPoool)  (None, 7, 7, 64)          0         
 flatten (Flatten)           (None, 3136)              0         
 dense (Dense)               (None, 128)               401536    
 dense_1 (Dense)             (None, 10)                1290      
=================================================================

```

##  Results

The model consistently achieves **~91-92% accuracy** on the test set within 10 epochs.

### Sample Predictions

Below are sample predictions from the test set. Green labels indicate a correct prediction, while red labels (if any) indicate a mismatch between the predicted and true category.
![Sample Predictions](assets/sample_predictions.png)
##  How to Use

1. Clone the repository:
```bash
git clone https://github.com/ibrahim-mosaad/cnn-fashion_mnist-classification.git

```


2. Install dependencies:
```bash
pip install tensorflow tensorflow-datasets matplotlib numpy

```


3. Run the notebook `cnn-fashion-mnist.ipynb` to train and evaluate.

##  Next Steps

* **Data Augmentation:** Implement flips and rotations to increase robustness.
* **Advanced Architectures:** Experiment with Dropout layers or Batch Normalization to reduce overfitting.
* **Deployment:** Export the `.keras` model to a TensorFlow Serving or Flask API.

---

*Created for learning and showcasing modern CNN pipelines.*
