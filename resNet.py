import tensorflow as tf
from tensorflow.keras import layers, models

# Residual Block
def residual_block(x, filters, kernel_size=3, stride=1, use_batch_norm=True):
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        if use_batch_norm:
            shortcut = layers.BatchNormalization()(shortcut)
    
    # Add the shortcut to the output
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# Build ResNet model
def resnet():
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Initial Conv layer
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Residual blocks with increasing filter sizes
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    
    x = residual_block(x, 128, stride=2)  # Downsample
    x = residual_block(x, 128)
    
    x = residual_block(x, 256, stride=2)  # Downsample
    x = residual_block(x, 256)
    
    x = residual_block(x, 512, stride=2)  # Downsample
    x = residual_block(x, 512)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(x)
    
    # Model definition
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices (One-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Instantiate the model
resnet_model = resnet()

# Compile the model
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
resnet_model.summary()

# Train the model
resnet_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
resnet_model.evaluate(x_test, y_test)
