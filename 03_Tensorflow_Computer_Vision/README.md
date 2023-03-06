---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Convolutional Neural Networks

CNNs are the ideal solution to discover pattern in visual data.


## Binary Image Classification

Using the __partial__ [Food101](https://www.tensorflow.org/datasets/catalog/food101) dataset:

```
mkdir datasets && cd datasets
wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip
unzip pizza_steak.zip && rm pizza_steak
```

```
tree -L 3
datasets
└── pizza_steak
    ├── test
    │   ├── pizza
    │   └── steak
    └── train
        ├── pizza
        └── steak
```


### Dependencies

```python
import itertools
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
```

### Inspect the Data

```python
# inspect data
training_data_dir = pathlib.Path("../datasets/pizza_steak/train")
testing_data_dir = pathlib.Path("../datasets/pizza_steak/test")
# create class names from sub dir names
class_names = np.array(sorted([item.name for item in training_data_dir.glob("*")]))

str(training_data_dir), str(testing_data_dir), str(class_names)
# ('../datasets/pizza_steak/train',
#  '../datasets/pizza_steak/test',
#  "['pizza' 'steak']")
```

```python
# display random images
def view_random_image(target_dir, target_class):
    target_folder = str(target_dir) + "/" + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(str(target_class) + str(img.shape))
    plt.axis("off")
    
    return tf.constant(img)

fig = plt.figure(figsize=(12, 12))
plot1 = fig.add_subplot(1, 2, 1)
pizza_image = view_random_image(target_dir = training_data_dir, target_class=class_names[0])
plot2 = fig.add_subplot(1, 2, 2)
steak_image = view_random_image(target_dir = training_data_dir, target_class=class_names[1])
plot1.title.set_text('Random Pizza Image')
plot2.title.set_text('Random Steak Image')

sample_image / 255

# the image is 512x384 pixels with 3 colour values per pixel
# to normalize the rgb values we need to divide all by 255

# <tf.Tensor: shape=(382, 512, 3), dtype=float32, numpy=
# array([[[0.8156863 , 0.7294118 , 0.87058824],
#         [0.827451  , 0.73333335, 0.8666667 ],
#         [0.8392157 , 0.73333335, 0.87058824],
#         ...,
#        [[0.75686276, 0.5254902 , 0.3529412 ],
#         [0.7411765 , 0.50980395, 0.3372549 ],
#         [0.7176471 , 0.4745098 , 0.3137255 ],
#         ...,
#         [0.59607846, 0.49803922, 0.38039216],
#         [0.6431373 , 0.5372549 , 0.42352942],
#         [0.65882355, 0.5529412 , 0.43529412]]], dtype=float32)>
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_01.png)

<!-- #region -->
### Building the Model

In machine learning, a classifier assigns a class label to a data point. For example, an image classifier produces a class label (e.g, pizza, steak) for what objects exist within an image. A convolutional neural network, or __CNN__ for short, is a type of classifier, which excels at solving this problem.

<br/><br/>

#### Rebuilding the Tiny VGG Architecture

_(see [CNN Explainer](https://poloclub.github.io/cnn-explainer/))_

* __Preprocessing__:

```python
tf.keras.utils.image_dataset_from_directory(
    directory,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    **kwargs
)
```

* If labels is `inferred`", the directory should contain subdirectories, each containing images for a class. Otherwise, the directory structure is ignored. "inferred" (labels are generated from the directory structure), None (no labels), or a list/tuple of integer labels of the same size as the number of image files found in the directory. Labels should be sorted according to the alphanumeric order of the image file paths
* String describing the encoding of labels. Options are:
  * `int`: means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
  * `categorical`: means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss).
  * `binary` means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy).
  * `None` (no labels).

<br/><br/>

__Conv2D Layer Options__

* __Filters__: How many filters should be applied to the input tensor (`10`, `32`, `64`, `128`).
* __Kernel Size__: Sets the filter size.
* __Padding__: `same` pads target tensor with zeros to preserve input shape. `valid` lowers the output shape.
* __Strides__: `strides=1` moves the filter across an image 1 pixel at a time.
<!-- #endregion -->

```python
seed = 42
batch_size = 32
img_height = 224
img_width = 224

tf.random.set_seed(seed)

# train and test data dirs
train_dir = "../datasets/pizza_steak/train/"
test_dir = "../datasets/pizza_steak/test/"

training_data = image_dataset_from_directory(train_dir,
                                              labels='inferred',
                                              label_mode='binary',
                                              seed=seed,
                                              image_size=(img_height, img_width),
                                              batch_size=batch_size)

testing_data = image_dataset_from_directory(test_dir,
                                              labels='inferred',
                                              label_mode='binary',
                                              seed=seed,
                                              image_size=(img_height, img_width),
                                              batch_size=batch_size)

# building the model
cnn_model = Sequential([
  Rescaling(1./255),
  Conv2D(filters=10, 
         kernel_size=3,
         activation="relu", 
         input_shape=(224, 224, 3)),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(pool_size=2, padding="same"),
  Conv2D(10, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(2),
  Flatten(),
  Dense(1, activation="sigmoid")
])

# compile the model
cnn_model.compile(loss="binary_crossentropy",
                 optimizer=Adam(learning_rate=1e-3),
                 metrics=["accuracy"])

# fitting the model
history_cnn = cnn_model.fit(training_data, epochs=5,
                            steps_per_epoch=len(training_data),
                            validation_data=testing_data,
                            validation_steps=len(testing_data))

# Found 1500 images belonging to 2 classes.
# Found 500 images belonging to 2 classes.
# Epoch 1/5

# Epoch 5/5
# 47/47 [==============================] - 2s 47ms/step - loss: 0.3347 - accuracy: 0.8540 - val_loss: 0.2927 - val_accuracy: 0.8820
```

```python
cnn_model.summary()

# Model: "sequential_8"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  rescaling (Rescaling)       (None, 224, 224, 3)       0 
#  conv2d_35 (Conv2D)          (None, 222, 222, 10)      280 
#  conv2d_36 (Conv2D)          (None, 220, 220, 10)      910 
#  max_pooling2d_17 (MaxPooling2D)  (None, 110, 110, 10) 0 
#  conv2d_37 (Conv2D)          (None, 108, 108, 10)      910 
#  conv2d_38 (Conv2D)          (None, 106, 106, 10)      910
#  max_pooling2d_18 (MaxPooling2D)  (None, 53, 53, 10)   0
#  flatten_8 (Flatten)         (None, 28090)             0
#  dense_8 (Dense)             (None, 1)                 28091 
# =================================================================
# Total params: 31,101
# Trainable params: 31,101
# Non-trainable params: 0
# _________________________________________________________________
```

### Building a Baseline Model

Above I already started with a CNN that was ideal for the given problem. Let's take a few steps back and try to work our way up to it by establishing a simple and fast baseline first. Fitting a machine learning model follows 3 steps:

1. Create a Baseline Model
2. Overfit a complexer model to improve the validation metric
  * Increase # of conv layers
  * Increas # of filters in conv layers
  * Add another dense layer above the output
3. Reduce the overfit
  * Add data augmentation
  * Add regularization layers (like pooling layers)
  * Add more, varied training data

```python
tf.random.set_seed(42)

model_cnn_base = Sequential([
  Rescaling(1./255),
  Conv2D(filters=10,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding="same",
          activation="relu",
          input_shape=(224, 224, 3),
          name="input_layer"),
  Conv2D(10, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"),
  Flatten(),
  Dense(1, activation="sigmoid", name="output_layer")
])

model_cnn_base.compile(loss="binary_crossentropy",
                      optimizer=Adam(learning_rate=(1e-3)),
                      metrics=["accuracy"])

history_cnn_baseline = model_cnn_base.fit(training_data, epochs=5,
                        steps_per_epoch=len(training_data),
                        validation_data=testing_data,
                        validation_steps=len(testing_data))

# Epoch 5/5
# 47/47 [==============================] - 3s 59ms/step - loss: 0.0901 - accuracy: 0.9740 - val_loss: 0.4827 - val_accuracy: 0.8020
```

#### Evaluating the Baseline Model

```python
# print loss curves
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_cnn.history).plot(ax=axes[0], title="Tiny VGG")
pd.DataFrame(history_cnn_baseline.history).plot(ax=axes[1], title="Baseline Model")

# as pointed out above - we can see that the validation loss for the baseline model
# stops improving. But the loss on the trainings data keeps falling
# => this points to our model __overfitting__ the training dataset
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_02.png)


#### Reducing the Overfit

Improve the evaluation metrics by tackling the overfitting issue:

```python
# adding pooling layers
# maxpool takes a square with size=poolsize (2 => 2x2)
# combines those pixel into 1 with the max value
# this looses fine details and directs your model
# towards the larger features in your image
model_cnn_base_pool = Sequential([
    Rescaling(1./255),
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 2, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 2, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model_cnn_base_pool.compile(loss="binary_crossentropy",
                           optimizer=Adam(learning_rate=1e-3),
                           metrics=["accuracy"])

history_cnn_baseline_pool = model_cnn_base_pool.fit(training_data, epochs=10,
                        steps_per_epoch=len(training_data),
                        validation_data=testing_data,
                        validation_steps=len(testing_data))

# Epoch 10/10
# 47/47 [==============================] - 1s 27ms/step - loss: 0.2748 - accuracy: 0.8907 - val_loss: 0.2552 - val_accuracy: 0.9040
```

```python
# print loss curves
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_cnn_baseline.history).plot(ax=axes[0], title="Baseline Model")
pd.DataFrame(history_cnn_baseline_pool.history).plot(ax=axes[1], title="Pooled Baseline Model")

# I increased the number of epochs to better see the result
# now it is obvious - adding pooling layers already solved the overfit
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_03.png)


Another tool we can use to improve the performance of an overfitting model is __Data Augmentation__.

```python
# to further generalize we could add more images that add variations
# but we get a similar effect from just modifying our training images
# randomly using augmentations to increase diversity

data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1),
    RandomContrast(0.2),
    RandomBrightness(factor=0.2)
])

model_cnn_base_pool_aug = Sequential([
    data_augmentation,
    Rescaling(1./255),
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 2, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 2, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model_cnn_base_pool_aug.compile(loss="binary_crossentropy",
                           optimizer=Adam(learning_rate=1e-3),
                           metrics=["accuracy"])

history_cnn_baseline_pool_aug = model_cnn_base_pool_aug.fit(training_data, epochs=50,
                        steps_per_epoch=len(training_data),
                        validation_data=testing_data,
                        validation_steps=len(testing_data))

# Epoch 50/50
# 47/47 [==============================] - 7s 145ms/step - loss: 0.3436 - accuracy: 0.8593 - val_loss: 0.2353 - val_accuracy: 0.9080
```

```python
# print loss curves
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_cnn_baseline_pool.history).plot(ax=axes[0], title="Pooled Baseline Model")
pd.DataFrame(history_cnn_baseline_pool_aug.history).plot(ax=axes[1], title="Augmented Baseline Model")

# adding too many data augmentation can lead to a degradation of the performance of a model

```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_04.png)


Add shuffle to our datasets:

```python
# randomize the order in which your models see the training images
# to remove biases in the order the data was collected in
seed = 42
batch_size = 32
img_height = 224
img_width = 224

tf.random.set_seed(seed)

# train and test data dirs
train_dir = "../datasets/pizza_steak/train/"
test_dir = "../datasets/pizza_steak/test/"

training_data_shuffled = image_dataset_from_directory(train_dir,
                                              labels='inferred',
                                              label_mode='binary',
                                              seed=seed,
                                              shuffle=True,
                                              image_size=(img_height, img_width),
                                              batch_size=batch_size)

testing_data_shuffled = image_dataset_from_directory(test_dir,
                                              labels='inferred',
                                              label_mode='binary',
                                              seed=seed,
                                              shuffle=True,
                                              image_size=(img_height, img_width),
                                              batch_size=batch_size)

# Found 1500 files belonging to 2 classes.
# Found 500 files belonging to 2 classes.
```

```python
# re-run the same pooled and augmented model as before on shuffled data
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1),
    RandomContrast(0.2),
    RandomBrightness(factor=0.2)
])

model_cnn_base_pool_aug_shuffle = Sequential([
    data_augmentation,
    Rescaling(1./255),
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 2, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 2, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model_cnn_base_pool_aug_shuffle.compile(loss="binary_crossentropy",
                           optimizer=Adam(learning_rate=1e-3),
                           metrics=["accuracy"])

history_cnn_baseline_pool_aug_shuffle = model_cnn_base_pool_aug_shuffle.fit(training_data_shuffled, epochs=50,
                        steps_per_epoch=len(training_data_shuffled),
                        validation_data=testing_data_shuffled,
                        validation_steps=len(testing_data_shuffled))

# Epoch 50/50
# 47/47 [==============================] - 8s 160ms/step - loss: 0.3257 - accuracy: 0.8580 - val_loss: 0.2617 - val_accuracy: 0.8860
```

```python
# print loss curves
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_cnn_baseline_pool_aug.history).plot(ax=axes[0], title="Augmented Baseline Model")
pd.DataFrame(history_cnn_baseline_pool_aug_shuffle.history).plot(ax=axes[1], title="Augmented Baseline Model (Shuffled)")

# the shuffled data shows a much steeper descent in the loss value:
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_05.png)

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_cnn.history).plot(ax=axes[0], title="Original Tiny VGG")
pd.DataFrame(history_cnn_baseline_pool_aug_shuffle.history).plot(ax=axes[1], title="Augmented Baseline Model (Shuffled)")
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_06.png)


Now that I have the data preprocessing dialed and am getting similar results to the initial __Tiny VGG__ run I want to see how this model performce now with the optimized data:

```python
# re-run the augmented data through the Tiny VGG architecture model
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1),
    RandomContrast(0.2),
    RandomBrightness(factor=0.2)
])

vgg_model = Sequential([
  data_augmentation,
  Rescaling(1./255),
  Conv2D(filters=10, 
         kernel_size=3,
         activation="relu", 
         input_shape=(224, 224, 3)),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(pool_size=2, padding="same"),
  Conv2D(10, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(2),
  Flatten(),
  Dense(1, activation="sigmoid")
])

vgg_model.compile(loss="binary_crossentropy",
                           optimizer=Adam(learning_rate=1e-3),
                           metrics=["accuracy"])

history_vgg_model = vgg_model.fit(training_data_shuffled, epochs=50,
                        steps_per_epoch=len(training_data_shuffled),
                        validation_data=testing_data_shuffled,
                        validation_steps=len(testing_data_shuffled))

# Epoch 50/50
# 47/47 [==============================] - 8s 175ms/step - loss: 0.2746 - accuracy: 0.8927 - val_loss: 0.2192 - val_accuracy: 0.9200
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_cnn_baseline_pool_aug_shuffle.history).plot(ax=axes[1], title="Augmented Baseline Model (Shuffled)")
pd.DataFrame(history_vgg_model.history).plot(ax=axes[0], title="Tiny VGG with Augumented Data")
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_07.png)


### Making Predictions on Custom Data

Now that I have a model that looks like it is performing well I can try to run a prediction on a personal picture from my favorite pizza place on the beach of Koh Rong in Cambodia:

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/pizza.jpg)

```python
pizza_path = "../assets/pizza.jpg"
pizza_img = mpimg.imread(pizza_path)
pizza_img.shape
# (426, 640, 3)
# before passing the image to our model we first need to pre-process it the same
# way we processed our training images.

steak_path = "../assets/steak.jpg"
```

```python
# helper function to pre-process images for predictions
def prepare_image(file_name, im_shape=224):
    # read in image
    img = tf.io.read_file(file_name)
    # image array => tensor
    img = tf.image.decode_image(img)
    # reshape to training size
    img = tf.image.resize(img, size=[im_shape, im_shape])
    # we don't need to normalize the image this is done by the model itself
    # img = img/255
    # add a dimension in front for batch size => shape=(1, 224, 224, 3)
    img = tf.expand_dims(img, axis=0)
    return img
```

```python
test_image_steak = prepare_image(file_name=steak_path)
test_image_pizza = prepare_image(file_name=pizza_path)
test_image_pizza
# the image now has the right shape to be ingested by our model:
# <tf.Tensor: shape=(1, 224, 224, 3), dtype=float32, numpy=
# array([[[[136.07143  , 141.07143  , 111.07143  ],
#          ...
```

```python
prediction_pizza = vgg_model.predict(test_image_pizza)
prediction_pizza
# we receive a prediction probability of `~0.86`
# array([[0.02831189]], dtype=float32)
```

```python
# make the propability output "readable"
pred_class_pizza = class_names[int(tf.round(prediction_pizza))]
pred_class_pizza
# 'pizza'
```

```python
prediction_steak = vgg_model.predict(test_image_steak)
pred_class_steak = class_names[int(tf.round(prediction_steak))]
pred_class_steak
# 'steak'
```

```python
# making the process a bit more visually appealing
def predict_and_plot(model, file_name, class_names):
    # load the image and preprocess
    img = prepare_image(file_name)
    # run prediction
    prediction = model.predict(img)
    # get predicted class name
    pred_class = class_names[int(tf.round(prediction))]
    # plot image & prediction
    plt.imshow(mpimg.imread(file_name))
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
```

```python
# a few more images to test with
pizza_path2 = "../assets/pizza2.jpg"
pizza_path3 = "../assets/pizza3.jpg"
steak_path2 = "../assets/steak2.jpg"
steak_path3 = "../assets/steak3.jpg"

predict_and_plot(model=vgg_model, file_name=pizza_path3, class_names=class_names)
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_08.png)


![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/nice.gif)


## Multiclass Image Classification

<!-- #region -->
* cd datasets
* wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip
* unzip 10_food_classes_all_data.zip && rm 10_food_classes_all_data.zip


```
10_food_classes_all_data
├── test
│   ├── chicken_curry
│   ├── chicken_wings
│   ├── fried_rice
│   ├── grilled_salmon
│   ├── hamburger
│   ├── ice_cream
│   ├── pizza
│   ├── ramen
│   ├── steak
│   └── sushi
└── train
    ├── chicken_curry
    ├── chicken_wings
    ├── fried_rice
    ├── grilled_salmon
    ├── hamburger
    ├── ice_cream
    ├── pizza
    ├── ramen
    ├── steak
    └── sushi

23 directories, 0 files
```

<!-- #endregion -->

### Visualizing the Data

```python
# set directories
training_directory = "../datasets/10_food_classes_all_data/train/"
testing_directory = "../datasets/10_food_classes_all_data/test/"

# get class names
data_dir = pathlib.Path(training_directory)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
len(class_names), class_names 

# the data set has 10 classes:
# (10,
# array(['chicken_curry', 'chicken_wings', 'fried_rice', 'grilled_salmon',
#        'hamburger', 'ice_cream', 'pizza', 'ramen', 'steak', 'sushi'],
#       dtype='<U14'))
```

```python
# visualizing the dataset
## display random images
def view_random_image(target_dir, target_class):
    target_folder = str(target_dir) + "/" + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(str(target_class) + str(img.shape))
    plt.axis("off")
    
    return tf.constant(img)

fig = plt.figure(figsize=(12, 6))
plot1 = fig.add_subplot(1, 2, 1)
plot1.title.set_text(f'Class: {class_names[0]}')
pizza_image = view_random_image(target_dir = training_directory, target_class=class_names[0])
plot2 = fig.add_subplot(1, 2, 2)
plot2.title.set_text(f'Class: {class_names[1]}')
steak_image = view_random_image(target_dir = training_directory, target_class=class_names[1])

fig = plt.figure(figsize=(12, 6))
plot3 = fig.add_subplot(1, 2, 1)
plot3.title.set_text(f'Class: {class_names[2]}')
pizza_image = view_random_image(target_dir = training_directory, target_class=class_names[2])
plot4 = fig.add_subplot(1, 2, 2)
plot4.title.set_text(f'Class: {class_names[3]}')
steak_image = view_random_image(target_dir = training_directory, target_class=class_names[3])

fig = plt.figure(figsize=(12, 6))
plot5 = fig.add_subplot(1, 2, 1)
plot5.title.set_text(f'Class: {class_names[4]}')
pizza_image = view_random_image(target_dir = training_directory, target_class=class_names[4])
plot6 = fig.add_subplot(1, 2, 2)
plot6.title.set_text(f'Class: {class_names[5]}')
steak_image = view_random_image(target_dir = training_directory, target_class=class_names[5])

fig = plt.figure(figsize=(12, 6))
plot7 = fig.add_subplot(1, 2, 1)
plot7.title.set_text(f'Class: {class_names[6]}')
pizza_image = view_random_image(target_dir = training_directory, target_class=class_names[6])
plot8 = fig.add_subplot(1, 2, 2)
plot8.title.set_text(f'Class: {class_names[7]}')
steak_image = view_random_image(target_dir = training_directory, target_class=class_names[7])
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_09.png)

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_10.png)

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_11.png)

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_12.png)


### Preprocessing the Data

```python
seed = 42
batch_size = 32
img_height = 224
img_width = 224

tf.random.set_seed(seed)

training_data_multi = image_dataset_from_directory(training_directory,
                                              labels='inferred',
                                              label_mode='categorical',
                                              seed=seed,
                                              shuffle=True,
                                              image_size=(img_height, img_width),
                                              batch_size=batch_size)

testing_data_multi = image_dataset_from_directory(testing_directory,
                                              labels='inferred',
                                              label_mode='categorical',
                                              seed=seed,
                                              shuffle=True,
                                              image_size=(img_height, img_width),
                                              batch_size=batch_size)
# Found 7500 files belonging to 10 classes.
# Found 2500 files belonging to 10 classes.
```

### Building the Model

```python
tf.random.set_seed(seed)
# building the model based on the tiny vgg architecture
vgg_model_multiclass = Sequential([
  Rescaling(1./255),
  Conv2D(filters=10, 
         kernel_size=3,
         activation="relu", 
         input_shape=(img_height, img_width, 3)),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(pool_size=2, padding="valid"),
  Conv2D(10, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(2, padding="valid"),
  Flatten(),
  Dense(len(class_names), activation="softmax")
])

# compile the model
vgg_model_multiclass.compile(loss="categorical_crossentropy",
                 optimizer=Adam(learning_rate=1e-3),
                 metrics=["accuracy"])

# fitting the model
history_vgg_model_multiclass = vgg_model_multiclass.fit(training_data_multi, epochs=5,
                            steps_per_epoch=len(training_data_multi),
                            validation_data=testing_data_multi,
                            validation_steps=len(testing_data_multi))

# Epoch 5/5
# 235/235 [==============================] - 12s 52ms/step - loss: 0.2465 - accuracy: 0.9251 - val_loss: 4.0673 - val_accuracy: 0.2760
```

```python
pd.DataFrame(history_vgg_model_multiclass.history).plot(title="Tiny VGG (Multiclass)")

# The training loss and accuracy are getting close to being perfect
# while the validation loss is running away => overfitting
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_13.png)


#### Reduce Overfitting

```python
tf.random.set_seed(seed)
# reduce the complexity of our model to minimize the overfit
vgg_model_multiclass_simplified = Sequential([
  Rescaling(1./255),
  Conv2D(filters=10, 
         kernel_size=3,
         activation="relu", 
         input_shape=(img_height, img_width, 3)),
  MaxPool2D(pool_size=2, padding="valid"),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(2, padding="valid"),
  Flatten(),
  Dense(len(class_names), activation="softmax")
])

# compile the model
vgg_model_multiclass_simplified.compile(loss="categorical_crossentropy",
                 optimizer=Adam(learning_rate=1e-3),
                 metrics=["accuracy"])

# fitting the model
history_vgg_model_multiclass_simplified = vgg_model_multiclass_simplified.fit(
                            training_data_multi, epochs=5,
                            steps_per_epoch=len(training_data_multi),
                            validation_data=testing_data_multi,
                            validation_steps=len(testing_data_multi))

# Epoch 5/5
# 235/235 [==============================] - 12s 52ms/step - loss: 0.2465 - accuracy: 0.9251 - val_loss: 4.0673 - val_accuracy: 0.2760
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_vgg_model_multiclass.history).plot(ax=axes[0], title="Tiny VGG (Multiclass)")
pd.DataFrame(history_vgg_model_multiclass_simplified.history).plot(ax=axes[1], title="Simple Tiny VGG (Multiclass)")
# that did not help at all...
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_14.png)

```python
tf.random.set_seed(seed)
# adding data augmentations
# i experimented a little bit with this
# things can go horrible wrong if you
# add too much :)
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
#     RandomZoom(0.1),
#     RandomContrast(0.2),
#     RandomBrightness(0.2)
])

# building the model
vgg_model_multiclass_augmentation = Sequential([
  data_augmentation,
  Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  Conv2D(16, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(pool_size=2, padding="same"),
  Conv2D(16, 3, activation="relu"),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(2, padding="same"),
  Flatten(),
  Dense(len(class_names), activation="softmax")
])

# compile the model
vgg_model_multiclass_augmentation.compile(loss="categorical_crossentropy",
                 optimizer=Adam(learning_rate=1e-3),
                 metrics=["accuracy"])

# fitting the model
history_vgg_model_multiclass_augmentation = vgg_model_multiclass_augmentation.fit(
                            training_data_multi, epochs=25,
                            steps_per_epoch=len(training_data_multi),
                            validation_data=testing_data_multi,
                            validation_steps=len(testing_data_multi))

# Epoch 10/10
# 235/235 [==============================] - 23s 99ms/step - loss: 1.6899 - accuracy: 0.4245 - val_loss: 1.8349 - val_accuracy: 0.3736
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_vgg_model_multiclass.history).plot(ax=axes[0], title="Tiny VGG (Multiclass)")
pd.DataFrame(history_vgg_model_multiclass_augmentation.history).plot(ax=axes[1], title="Augmented Tiny VGG (Multiclass)")
# this looks already alot better - but the improvement is very slow...
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_15.png)

```python
# try adding a few more epochs to get those curves closer
history_vgg_model_multiclass_augmentation = vgg_model_multiclass_augmentation.fit(
                            training_data_multi, epochs=25,
                            steps_per_epoch=len(training_data_multi),
                            validation_data=testing_data_multi,
                            validation_steps=len(testing_data_multi))

# as expected - running the training for longer - slowly - improves the accuracy
# as well as validation accuracy for the model. The validation loss, however, remains
# stubbornly high:
# Epoch 25/25
# 235/235 [==============================] - 23s 97ms/step - loss: 1.3810 - accuracy: 0.5397 - val_loss: 1.7875 - val_accuracy: 0.4048
```

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
pd.DataFrame(history_vgg_model_multiclass.history).plot(ax=axes[0], title="Tiny VGG (Multiclass)")
pd.DataFrame(history_vgg_model_multiclass_augmentation.history).plot(ax=axes[1], title="Augmented Tiny VGG (Multiclass)")
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_16.png)


### Making Predictions

```python
img_height = 224
img_width = 224

# helper function to pre-process images for predictions
def prepare_image(file_name, img_height=img_height, img_width=img_width):
    # read in image
    img = tf.io.read_file(file_name)
    # image array => tensor
    img = tf.image.decode_image(img)
    # reshape to training size
    img = tf.image.resize(img, size=[img_height, img_width])
    # we don't need to normalize the image this is done by the model itself
    # img = img/255
    # add a dimension in front for batch size => shape=(1, 224, 224, 3)
    img = tf.expand_dims(img, axis=0)
    return img

# adapt plot function for multiclass predictions
def predict_and_plot_multi(model, file_name, class_names):
    # load the image and preprocess
    img = prepare_image(file_name)
    # run prediction
    prediction = model.predict(img)
    # check for multiclass
    if len(prediction[0]) > 1:
        # pick class with highest probability
        pred_class = class_names[tf.argmax(prediction[0])]
    else:
        # binary classifications only return 1 probability value
        pred_class = class_names[int(tf.round(prediction))]
    # plot image & prediction
    plt.imshow(mpimg.imread(file_name))
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)
```

```python
fig = plt.figure(figsize=(12, 6))
plot7 = fig.add_subplot(1, 2, 1)
pizza_image = predict_and_plot_multi(model=vgg_model_multiclass_augmentation, file_name=pizza_path, class_names=class_names)
plot8 = fig.add_subplot(1, 2, 2)
steak_image = predict_and_plot_multi(model=vgg_model_multiclass_augmentation, file_name=steak_path, class_names=class_names)
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_17.png)

```python
# a few more images to test with
ice_cream_path = "../assets/ice_cream.jpg"
hamburger_path = "../assets/hamburger.jpg"

fig = plt.figure(figsize=(12, 6))
plot7 = fig.add_subplot(1, 2, 1)
pizza_image = predict_and_plot_multi(model=vgg_model_multiclass_augmentation, file_name=ice_cream_path, class_names=class_names)
plot8 = fig.add_subplot(1, 2, 2)
steak_image = predict_and_plot_multi(model=vgg_model_multiclass_augmentation, file_name=hamburger_path, class_names=class_names)
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_18.png)

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/hmmmm.gif)

<!-- #region -->
This pretty much sums up the `val_accuracy: 0.4048` value I gut during the training. With 10 I would get an ~ accuracy of `10%` if the model was guessing randomly. The training got us factor 4 improvement. But it is far from perfect.


The validation accuracy was still increasing - so if I kept training the model I would get better results. But the increase was very slow - it might take a long time. The validation loss was running away in the beginning. But adding image augmentations pulled it back down nicely. I have been experimenting with different augmentations and their effect ranged from good to terrible :) - there is still some room for improvement adding or adjusting augmentations.

The next thing - or maybe the first - would be to consolidate the dataset. Some of the images in it are terrible. They have plenty of clutter in the background. There is grilled salmon that looks like sushi. Or close-up chicken curry that could be a pizza. Removing some of those images, or preferably replacing them with higher quality images, will improve our model performance.

We could also check out the confuion matrix to see if there are classes that perform particularly bad. If there are we can concentrate our efforts on them. (see below - `sushi` and `ramen` seems to perform _OK_)

<!-- #endregion -->

### Saving & Loading the Model

```python
# save a model
vgg_model_multiclass_augmentation.save("../saved_models/vgg_model_multiclass_augmentation")

# INFO:tensorflow:Assets written to: ../saved_models/vgg_model_multiclass_augmentation/assets
```

```python
# load a trained model
loaded_model = tf.keras.models.load_model("../saved_models/vgg_model_multiclass_augmentation")
```

```python
# test if the model was loaded successful
sushi_path = "../assets/sushi.jpg"
ramen_path = "../assets/ramen.jpg"

fig = plt.figure(figsize=(12, 6))
plot7 = fig.add_subplot(1, 2, 1)
pizza_image = predict_and_plot_multi(model=loaded_model, file_name=sushi_path, class_names=class_names)
plot8 = fig.add_subplot(1, 2, 2)
steak_image = predict_and_plot_multi(model=loaded_model, file_name=ramen_path, class_names=class_names)
```

![Convolutional Neural Networks](https://github.com/mpolinowski/tf-2023/blob/master/assets/03_Tensorflow_Convolutional_Neural_Networks_19.png)


To fast forward -> Next step __Transfer Learning__