#  Basic Image Classification with TensorFlow (MNIST)

This project demonstrates how to build, train, and evaluate a neural network using **TensorFlow and Keras** to classify handwritten digit images from the **MNIST dataset**.

The goal is to predict which digit (0–9) appears in a grayscale image of size 28×28 pixels.

---

##  Project Overview

- **Task:** Multi-class image classification  
- **Dataset:** MNIST handwritten digits  
- **Framework:** TensorFlow with Keras API  
- **Model Type:** Artificial Neural Network (Dense layers)  
- **Evaluation Metric:** Accuracy  

---

##  Dataset Details

The MNIST dataset contains:

- **60,000 training images**
- **10,000 test images**
- Image size: **28 × 28**
- Image type: **Grayscale**
- Classes: **10 digits (0–9)**

The dataset is automatically loaded using TensorFlow’s built-in dataset loader.

---

##  Preprocessing Steps

Before training, the following preprocessing operations were applied:

###  Normalization
Pixel values originally ranged from 0 to 255.  
They were scaled to the range **0–1** to make training stable and faster.

###  Flattening
Images were reshaped from:

28 × 28 → 784


This was required because Dense layers expect one-dimensional input.

###  One-Hot Encoding
Digit labels were converted into vectors of length 10 to match the softmax output layer and categorical crossentropy loss function.

---

##  Model Architecture

The neural network is built using a Sequential model with the following layers:

1. **Dense Layer (128 neurons, ReLU activation)**  
   - Learns patterns in pixel values  
   - Adds non-linearity  

2. **Dense Output Layer (10 neurons, Softmax activation)**  
   - Produces probability scores for each digit  

---

##  Model Compilation

The model was compiled using:

- **Optimizer:** Stochastic Gradient Descent (SGD)  
- **Loss Function:** Categorical Crossentropy  
- **Metric:** Accuracy  

---

##  Training

The model was trained using:

- Normalized training images
- One-hot encoded labels
- Multiple epochs

Training was performed using `model.fit()` to update the weights through backpropagation and gradient descent.

---

##  Evaluation

After training, the model was evaluated on the test dataset using:

model.evaluate(x_test, y_test)


This produced:

- **Test Accuracy**
- **Test Loss**

These values indicate how well the model generalizes to unseen images.

---

##  Making Predictions

The trained model outputs probabilities for each digit class.

To obtain the final predicted digit, the index of the highest probability is selected using:

argmax()


---

##  Key Concepts Learned

- Image classification
- Supervised learning
- Neural networks
- Dense layers
- Activation functions (ReLU, Softmax)
- One-hot encoding
- Gradient descent
- Optimizers
- Training vs testing data
- Accuracy vs loss

---

##  Possible Improvements

Some ways this project could be extended:

- Use Convolutional Neural Networks (CNNs)
- Add more hidden layers
- Apply data augmentation
- Tune hyperparameters
- Analyze misclassified examples
- Add confusion matrix

---

##  Conclusion

This project provides a complete beginner-friendly pipeline for image classification using TensorFlow. It introduces fundamental deep-learning concepts such as preprocessing, model construction, training, evaluation, and prediction, forming a strong foundation for more advanced computer-vision tasks.

