# Satellite Image Classification using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) for classifying satellite images into four categories: cloudy, desert, green area, and water.

## Table of Contents
- [Satellite Image Classification using Convolutional Neural Networks](#satellite-image-classification-using-convolutional-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Usage](#usage)
    - [Model Architecture](#model-architecture)
    - [The dataset is available at :](#the-dataset-is-available-at-)

## Installation

To run this project, you need to have the following dependencies installed:

- Python 3.x
- NumPy
- TensorFlow
- Keras

You can install the required packages using pip:
```python
pip install numpy tensorflow keras
```

## Dataset

The dataset used in this project consists of satellite images categorized into four classes:

1. Cloudy
2. Desert
3. Green Area
4. Water

The images are 64x64 pixels in size and have 3 color channels (RGB). The dataset is loaded from a local 'data' directory.

## Usage

1. Clone this repository to your local machine.
2. Ensure you have all the required dependencies installed.
3. Organize your dataset in the 'data' directory with subdirectories for each class.
4. Run the script:

```python
# Loading the dataset
data = 'data'
class_labels = ['cloudy', 'desert', 'green_area', 'water']
labels = {}
for i in range(len(class_labels)):
    labels[i] = class_labels[i]
x = []
y = []

# Add your data loading code here

# Building the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(len(class_labels), activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the Model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Model Architecture
The CNN model architecture is as follows:

Convolutional layer (32 filters, 3x3 kernel)
Max pooling layer (2x2)
Flatten layer
Dense layer (output layer with softmax activation)

The model is compiled using the Adam optimizer and Sparse Categorical Crossentropy loss function.
Results
The model's performance can be evaluated using the history object returned by the fit method. You can plot the training and validation accuracy and loss over epochs to visualize the model's learning progress.

### The dataset is available at :
https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification