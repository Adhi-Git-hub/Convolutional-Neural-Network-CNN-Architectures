import tensorflow as tf
from tensorflow.keras import layers, models

# Dense Block
def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        x = conv_block(x, growth_rate)
    return x

# Convolutional Block inside Dense Block
def conv_block(x, growth_rate):
    # Batch Normalization, ReLU, and 1x1 Conv
    x1 = layers.BatchNormalization()(x)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(4 * growth_rate, (1, 1), padding='same')(x1)
    
    # Batch Normalization, ReLU, and 3x3 Conv
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv2D(growth_rate, (3, 3), padding='same')(x1)
    
    # Concatenate the input and output 
    x = layers.concatenate([x, x1])
    
    return x

# Transition Layer 
def transition_layer(x, compression):
    # Batch Normalization, ReLU, and 1x1 Conv
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(int(x.shape[-1] * compression), (1, 1), padding='same')(x)
    
    # Average Pooling to reduce the spatial dimensions
    x = layers.AveragePooling2D((2, 2), strides=2, padding='same')(x)
    
    return x

# DenseNet Model
def densenet():
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Initial Convolution
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    
    # Dense Block 1
    x = dense_block(x, num_layers=6, growth_rate=12)
    x = transition_layer(x, compression=0.5)
    
    # Dense Block 2
    x = dense_block(x, num_layers=12, growth_rate=12)
    x = transition_layer(x, compression=0.5)
    
    # Dense Block 3
    x = dense_block(x, num_layers=24, growth_rate=12)
    x = transition_layer(x, compression=0.5)
    
    # Dense Block 4
    x = dense_block(x, num_layers=16, growth_rate=12)
    
    # Global Average Pooling
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Output Layer
    output_layer = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


densenet_model = densenet()

# Compile the model
densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
densenet_model.summary()

# Train the model
densenet_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
densenet_model.evaluate(x_test, y_test)
