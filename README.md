# Handwritten Digit Recognition
This repository contains a Python script for training, evaluating, and using a deep learning model to recognize handwritten digits using the MNIST dataset.

## Requirements
Python 3.x
TensorFlow
NumPy
Matplotlib
OpenCV (cv2)
## Installation
### 1. Clone the repository:


git clone https://github.com/your_username/handwritten-digit-recognition.git
### 2. Install the required dependencies:


pip install tensorflow numpy matplotlib opencv-python
### 3. Download the MNIST dataset:
The dataset is automatically downloaded when running the script for the first time.

## Usage
### 1. Train the model:
Run the script train_model.py to train the model using the MNIST dataset.
The model architecture includes flatten and dense layers with dropout regularization.
Adjust the number of epochs as needed in the script (epochs = 10 by default).

### 2. Evaluate the model:
After training, the script prints the test loss and accuracy of the trained model.
These metrics indicate the performance of the model on unseen test data.

### 3. Predict handwritten digits:
Place the digit images to be predicted in the digits directory.
Run the script predict_digits.py to predict the handwritten digits.
The script processes each image, passes it through the trained model, and displays the predicted digit along with the image.
The predicted digit is printed on the console.
Ensure that OpenCV (cv2) and the necessary dependencies are installed to process the images.

## Results
The trained model achieves a high accuracy on the test set.
The prediction accuracy for the handwritten digit images is printed on the console.
The predicted digits along with the respective images are displayed using Matplotlib.
