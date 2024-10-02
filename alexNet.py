import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Load ImageNet dataset
# For this example, replace with actual ImageNet loading procedure
# You might use `image_dataset_from_directory` for custom loading
train_dataset = image_dataset_from_directory(
    "path",
    image_size=(224, 224),
    batch_size=128
)
val_dataset = image_dataset_from_directory(
    "path",
    image_size=(224, 224),
    batch_size=128
)

# Normalize the data (rescale to 0-1)
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess_image)
val_dataset = val_dataset.map(preprocess_image)

# Model definition
def alexnet():
    model = models.Sequential()

    # First Convolutional Layer (ImageNet)
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Third Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # Fourth Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # Fifth Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Fully Connected Layer 2
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer (1000 classes for ImageNet)
    model.add(layers.Dense(1000, activation='softmax'))

    return model

alexnet_model = alexnet()

# Compile the model 
alexnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
alexnet_model.summary()

# Train the model
alexnet_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model
alexnet_model.evaluate(val_dataset)
