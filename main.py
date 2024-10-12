import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

IMG_HEIGHT, IMG_WIDTH = 128, 128 # Define image dimensions


def load_image(image_path): # Function to load and preprocess images
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = img_to_array(img)  # Convert the image to a numpy array
    img = img / 255.0  # Normalize to [0, 1]
    return img

def load_label(label_path): # Function to load the labels from the text files
    with open(label_path, 'r') as f:
        label = int(f.read().strip())  # Read the label and convert to int
    return label

# Directory paths
image_dir = 'D:/robomaster/train/'
label_dir = 'D:/robomaster/train/'

# Initialize lists for storing images and labels
images = []
labels = []

# Iterate through the images folder
for image_filename in os.listdir(image_dir):
    if image_filename.endswith('.png'):
        image_path = os.path.join(image_dir, image_filename)
        
        # Get the corresponding text filename for the label
        label_filename = image_filename.replace('.png', '.txt')
        label_path = os.path.join(label_dir, label_filename)
        
        # Load and preprocess the image and the label
        img = load_image(image_path)
        label = load_label(label_path)
        
        # Append them to lists
        images.append(img)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create TensorFlow datasets for better performance
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Data augmentation (optional, if the dataset is small)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# Apply augmentation on training data
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# Batch and prefetch for performance
BATCH_SIZE = 32
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def create_lightweight_model(input_shape):
    model = models.Sequential()
    
    # First Conv Block
    model.add(layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.DepthwiseConv2D((3, 3), padding='same'))
    model.add(layers.Conv2D(32, (1, 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Conv Block
    model.add(layers.DepthwiseConv2D((3, 3), padding='same'))
    model.add(layers.Conv2D(64, (1, 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Squeeze-and-Excitation Layer
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(32, activation='relu'))
    
    # Classification Layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = create_lightweight_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Train the model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=10)