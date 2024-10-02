# Convolutional-Neural-Network-CNN-Architectures
Convolutional Neural Network(CNN) is a neural network architecture in Deep Learning, used to recognize the pattern from structured arrays. However, over many years, CNN architectures have evolved. Many variants of the fundamental CNN Architecture This been developed, leading to amazing advances in the growing deep-learning field.

## What is a CNN?

A Convolutional Neural Network (CNN) is a deep learning model used primarily for image classification, object detection, and other tasks that involve grid-like data, such as images. CNNs automatically detect important features in images, like edges and textures, without the need for manual feature engineering. They consist of layers like convolution, pooling, and fully connected layers to progressively learn image representations.

## Models in this Repository

### 1. LeNet
LeNet was one of the earliest CNN architectures and was designed for digit recognition. It was originally used to classify handwritten digits in the MNIST dataset. The architecture is simple and consists of two convolutional layers followed by two fully connected layers.

### 2. AlexNet
AlexNet was the breakthrough model that won the ImageNet competition in 2012. It uses five convolutional layers and three fully connected layers. AlexNet introduced the use of ReLU activations and dropout to reduce overfitting.

### 3. ResNet (Residual Networks)
ResNet is a deep CNN that introduces shortcut connections (or skip connections) to tackle the problem of vanishing gradients in very deep networks. This allows it to build extremely deep models like ResNet-50, which can perform better on large-scale image classification tasks.

### 4. GoogleNet (Inception)
GoogleNet, also known as Inception, introduced the concept of Inception modules, which allow the network to choose between multiple convolutional filter sizes at each layer. It was another significant improvement in the ImageNet competition, as it achieved high accuracy with relatively fewer parameters than AlexNet.

### 5. DenseNet
DenseNet connects each layer to every other layer in a feed-forward manner. This helps in strengthening feature propagation and reducing the number of parameters. DenseNet has been effective in tasks like image classification by utilizing features more efficiently.

## Datasets Used

- **CIFAR-10/CIFAR-100**: These are datasets of tiny 32x32 color images. CIFAR-10 has 10 classes, while CIFAR-100 has 100 classes.
- **MNIST**: This is a dataset of handwritten digits, consisting of grayscale 28x28 pixel images.

