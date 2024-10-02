import tensorflow as tf
from tensorflow.keras import layers, models

# Inception Module
def inception_module(x, f1, f3_in, f3, f5_in, f5, proj):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(x)
    
    # 1x1 conv -> 3x3 conv
    conv3 = layers.Conv2D(f3_in, (1, 1), padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(f3, (3, 3), padding='same', activation='relu')(conv3)
    
    # 1x1 conv -> 5x5 conv
    conv5 = layers.Conv2D(f5_in, (1, 1), padding='same', activation='relu')(x)
    conv5 = layers.Conv2D(f5, (5, 5), padding='same', activation='relu')(conv5)
    
    # 3x3 max pooling -> 1x1 conv
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = layers.Conv2D(proj, (1, 1), padding='same', activation='relu')(pool)
    
    # Concatenate all filters outputs
    output = layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
    
    return output

# Build GoogleNet model for CIFAR-10
def googlenet():
    input_layer = layers.Input(shape=(32, 32, 3))
    
    # Initial layers
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # Inception modules
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    
    # Auxiliary classifier 1
    aux1 = layers.AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux1 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(aux1)
    aux1 = layers.Flatten()(aux1)
    aux1 = layers.Dense(1024, activation='relu')(aux1)
    aux1 = layers.Dropout(0.7)(aux1)
    aux1 = layers.Dense(10, activation='softmax')(aux1)
    
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    
    # Auxiliary classifier 2
    aux2 = layers.AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux2 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(aux2)
    aux2 = layers.Flatten()(aux2)
    aux2 = layers.Dense(1024, activation='relu')(aux2)
    aux2 = layers.Dropout(0.7)(aux2)
    aux2 = layers.Dense(10, activation='softmax')(aux2)
    
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = inception_module(x, 256, 160, 320, 48, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout and final dense layer
    x = layers.Dropout(0.4)(x)
    output_layer = layers.Dense(10, activation='softmax')(x)
    
    # Model definition
    model = models.Model(inputs=input_layer, outputs=[output_layer, aux1, aux2])
    
    return model

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Instantiate the model
googlenet_model = googlenet()

# Compile the model
googlenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
googlenet_model.summary()

# Train the model
googlenet_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
googlenet_model.evaluate(x_test, y_test)
