# Convolutional-Neural-Network-CNN-Architectures
Convolutional Neural Network(CNN) is a neural network architecture in Deep Learning, used to recognize the pattern from structured arrays. However, over many years, CNN architectures have evolved. Many variants of the fundamental CNN Architecture This been developed, leading to amazing advances in the growing deep-learning field.

## Deep Learning Architectures with CIFAR and MNIST

This repository contains implementations of several popular deep learning architectures: LeNet, AlexNet, ResNet, GoogleNet, and DenseNet. These models have been trained on the CIFAR and MNIST datasets, providing hands-on experience with different convolutional neural networks (CNNs).

## What is a CNN?

A Convolutional Neural Network (CNN) is a specialized type of neural network used primarily for image data. CNNs consist of layers that automatically learn spatial hierarchies of features from input images, making them powerful for visual tasks.

### Key Concepts in CNNs

- **Filters (Kernels):** Small matrix filters are applied to the input data to extract features such as edges, textures, and more complex patterns. These filters slide over the image and detect different characteristics at each position.
  
- **Stride:** This refers to how the filter moves across the input. A stride of 1 means the filter moves one pixel at a time, while a larger stride means the filter skips pixels, reducing the size of the output.

- **Padding:** When the filter slides over the input, padding is sometimes added around the edges of the input to ensure that the output retains useful boundary information.

- **Max Pooling:** A downsampling operation that reduces the dimensionality of feature maps by taking the maximum value within a defined window, helping to reduce computation and control overfitting.

### Architectures in This Repository

#### LeNet
- **Description:** One of the earliest CNN models designed for handwritten digit recognition (MNIST dataset).
- **Key Features:** LeNet has two convolutional layers followed by subsampling (pooling) layers, and two fully connected layers.
  
#### AlexNet
- **Description:** AlexNet popularized deep learning for image recognition by winning the ImageNet competition in 2012. It uses large convolutional filters and multiple fully connected layers.
- **Key Features:** Utilizes five convolutional layers with large filters and high depth. It applies max pooling to downsample the input, and uses dropout in fully connected layers to reduce overfitting.

#### ResNet
- **Description:** ResNet (Residual Networks) introduced the idea of skip connections, allowing gradients to flow more easily through deep networks, thus addressing the vanishing gradient problem.
- **Key Features:** Incorporates identity connections, or "residuals," to bypass one or more layers, which enables extremely deep networks (50, 101, or more layers).

#### GoogleNet (Inception)
- **Description:** GoogleNet uses the Inception module, which performs multiple convolutions and pooling in parallel, combining the outputs. This model is highly efficient in terms of the number of parameters.
- **Key Features:** Introduces inception blocks that allow the network to capture different aspects of the input simultaneously. It has auxiliary classifiers that help in the learning process during training.

#### DenseNet
- **Description:** DenseNet improves information flow by connecting each layer to every other layer in a feed-forward fashion. It reduces the number of parameters by reusing feature maps.
- **Key Features:** Dense connections between layers lead to feature reuse, making the model efficient and effective at learning deep representations.


