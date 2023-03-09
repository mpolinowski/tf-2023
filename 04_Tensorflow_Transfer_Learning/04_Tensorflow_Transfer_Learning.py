import datetime
import itertools
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

# Check if GPU is Available
print('INFO :: ' + str(tf.config.list_physical_devices('GPU')))

# Set Global Variables
SEED = 42
BATCH_SIZE = 16
IMG_SHAPE = (224, 224)

# Set Directories for Training and Testing Dataset
training_directory = "../datasets/10_food_classes_10_percent/train/"
testing_directory = "../datasets/10_food_classes_10_percent/test/"

## get class names
data_dir = pathlib.Path(training_directory)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))

## check number of classes and class names
print('INFO :: Number of classes: ' + str(len(class_names)))
print('INFO :: Class names ' + str(class_names))


# Visualizing the Dataset
## pick random image from directory
def view_random_image(target_dir, target_class):
    target_folder = str(target_dir) + "/" + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(str(target_class) + str(img.shape))
    plt.axis("off")
    
    return tf.constant(img)


## Display 2 Images from 2 Random Classes
fig = plt.figure(figsize=(12, 6))
plot1 = fig.add_subplot(1, 2, 1)
plot1.title.set_text(f'Class: {class_names[0]}')
pizza_image = view_random_image(target_dir = training_directory, target_class=class_names[random.randrange(10)])
plot2 = fig.add_subplot(1, 2, 2)
plot2.title.set_text(f'Class: {class_names[1]}')
steak_image = view_random_image(target_dir = training_directory, target_class=class_names[random.randrange(10)])
plt.show()



# Load Training and Testing Dataset
training_data = image_dataset_from_directory(training_directory,
                                              labels='inferred',
                                              label_mode='categorical',
                                              seed=SEED,
                                              shuffle=True,
                                              image_size=IMG_SHAPE,
                                              batch_size=BATCH_SIZE)

testing_data = image_dataset_from_directory(testing_directory,
                                              labels='inferred',
                                              label_mode='categorical',
                                              seed=SEED,
                                              shuffle=True,
                                              image_size=IMG_SHAPE,
                                              batch_size=BATCH_SIZE)


# this time we only work with 10% of the training dataset
# Found 750 files belonging to 10 classes.
# Found 2500 files belonging to 10 classes.


# Tensorboard
## create a callback to track experiments in TensorBoard
def create_tensorboard_callback(dir_name, experiment_name):
    # log progress to log directory
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"Saving TensorBoard Log to: {log_dir}")
    return tensorboard_callback


# Select Pre-trained Models
## models to compare
resnet_model_url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5'
efficientnet_model_url = 'https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1'


# Model Generator Function
## add image augmentations (optional)
data_augmentation_layer = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1),
    RandomContrast(0.2),
    RandomBrightness(0.2)
])

## helper function to create a model
def create_model(model_url, num_classes):
    # download pre-trained model as a keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name='feature_extractor_layer')
    
    ### create sequential model
    model = tf.keras.Sequential([
        Rescaling(1./255, input_shape=IMG_SHAPE+(3,)),
        # data_augmentation_layer,
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])

    return model


# Creating a ResNet-based Model

resnet_model = create_model(resnet_model_url, len(class_names))

resnet_model.compile(loss="categorical_crossentropy",
                     optimizer=Adam(learning_rate=1e-3),
                     metrics=["accuracy"])

print(resnet_model.summary())


# Training the ResNet Model

tf.random.set_seed(SEED)

## fitting the model
history_resnet_model = resnet_model.fit(training_data, epochs=5,
                            steps_per_epoch=len(training_data),
                            validation_data=testing_data,
                            validation_steps=len(testing_data))