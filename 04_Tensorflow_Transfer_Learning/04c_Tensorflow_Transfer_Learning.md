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

# Transfer Learning

## Model Scaling

1. Pretraining EfficientNetB0 (10% of Dataset with 10 Classes)
2. Fine-Tuning (100% of Dataset with 10 Classes)
3. Scaling the Model to all 101 Classes (10% of Dataset with 101 Classes)

```python
import datetime
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
```

```python
# export helper functions from above into helper script
from helper import (create_tensorboard_callback,
                    create_checkpoint_callback,
                    plot_accuracy_curves,
                    combine_training_curves,
                    data_augmentation_layer_no_rescaling)
```

```python
# global variables
SEED = 42
BATCH_SIZE = 32
IMG_SHAPE = (224, 224)
```

### Get Data

```python
# preparing datasets
# wget https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip

training_directory = "../datasets/101_food_classes_10_percent/train"
testing_directory = "../datasets/101_food_classes_10_percent/test"

training_data_101_10 = image_dataset_from_directory(training_directory,
                                              labels='inferred',
                                              label_mode='categorical',
                                              seed=SEED,
                                              shuffle=True,
                                              image_size=IMG_SHAPE,
                                              batch_size=BATCH_SIZE)

testing_data_101_10 = image_dataset_from_directory(testing_directory,
                                              labels='inferred',
                                              label_mode='categorical',
                                              seed=SEED,
                                              shuffle=False,
                                              image_size=IMG_SHAPE,
                                              batch_size=BATCH_SIZE)

# get class names
class_names_101_10 = training_data_101_10.class_names
len(class_names_101_10), class_names_101_10

# Found 7575 files belonging to 101 classes.
# Found 25250 files belonging to 101 classes.

# (101,
#  ['apple_pie',
#   'baby_back_ribs',
#   'baklava',
#   'beef_carpaccio',
#   'beef_tartare',
#   'beet_salad',
#   'beignets',
#   'bibimbap',
#   'bread_pudding',
#   'breakfast_burrito',
#   'bruschetta',
#   'caesar_salad',
#   'cannoli',
#   'caprese_salad',
#   'carrot_cake',
#   'ceviche',
#   'cheese_plate',
#   'cheesecake',
#   'chicken_curry',
#   'chicken_quesadilla',
#   'chicken_wings',
#   'chocolate_cake',
#   'chocolate_mousse',
#   'churros',
#   'clam_chowder',
#   'club_sandwich',
#   'crab_cakes',
#   'creme_brulee',
#   'croque_madame',
#   'cup_cakes',
#   'deviled_eggs',
#   'donuts',
#   'dumplings',
#   'edamame',
#   'eggs_benedict',
#   'escargots',
#   'falafel',
#   'filet_mignon',
#   'fish_and_chips',
#   'foie_gras',
#   'french_fries',
#   'french_onion_soup',
#   'french_toast',
#   'fried_calamari',
#   'fried_rice',
#   'frozen_yogurt',
#   'garlic_bread',
#   'gnocchi',
#   'greek_salad',
#   'grilled_cheese_sandwich',
#   'grilled_salmon',
#   'guacamole',
#   'gyoza',
#   'hamburger',
#   'hot_and_sour_soup',
#   'hot_dog',
#   'huevos_rancheros',
#   'hummus',
#   'ice_cream',
#   'lasagna',
#   'lobster_bisque',
#   'lobster_roll_sandwich',
#   'macaroni_and_cheese',
#   'macarons',
#   'miso_soup',
#   'mussels',
#   'nachos',
#   'omelette',
#   'onion_rings',
#   'oysters',
#   'pad_thai',
#   'paella',
#   'pancakes',
#   'panna_cotta',
#   'peking_duck',
#   'pho',
#   'pizza',
#   'pork_chop',
#   'poutine',
#   'prime_rib',
#   'pulled_pork_sandwich',
#   'ramen',
#   'ravioli',
#   'red_velvet_cake',
#   'risotto',
#   'samosa',
#   'sashimi',
#   'scallops',
#   'seaweed_salad',
#   'shrimp_and_grits',
#   'spaghetti_bolognese',
#   'spaghetti_carbonara',
#   'spring_rolls',
#   'steak',
#   'strawberry_shortcake',
#   'sushi',
#   'tacos',
#   'takoyaki',
#   'tiramisu',
#   'tuna_tartare',
#   'waffles'])
```

### Model Building and Training

```python
# create callbacks from helper.py
## checkpoint callback
checkpoint_callback = create_checkpoint_callback(
    dir_name='../checkpoints/transfer_learning_scaling',
    experiment_name='101_classes_10_percent_dataset')

## tensorboard callback
tensorboard_callback = create_tensorboard_callback(
    dir_name='../tensorboard/transfer_learning_scaling',
    experiment_name='101_classes_10_percent_dataset')
```

```python
# get base model from keras applications
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
    include_top=False
)
base_model.trainable = False

# build model using keras functional API
input_layer = tf.keras.layers.Input(shape=IMG_SHAPE+(3,), name='input_layer')
# use image augmentation layer from helper.py
data = data_augmentation_layer_no_rescaling(input_layer)
# run in inference mode so batchnorm statistics don't get updated
# even after unfreezing the base model for fine-tuning
data = base_model(data, training=False)
data = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(data)
output_layer = tf.keras.layers.Dense(len(class_names_101_10), activation="softmax", name="output_layer")(data)

model = tf.keras.Model(input_layer, output_layer)

# compile the model
model.compile(loss='categorical_crossentropy',
               optimizer=tf.keras.optimizers.Adam(),
               metrics=['accuracy'])
```

```python
# fit the model
## training epochs before fine-tuning
pretraining_epochs = 5

history_model = model.fit(
                training_data_101_10,
                epochs=pretraining_epochs,
                steps_per_epoch=len(training_data_101_10),
                validation_data=testing_data_101_10,
                # evaluate performance on 15% of the testing dataset
                validation_steps=int(0.15 * len(testing_data_101_10)),
                callbacks=[tensorboard_callback,
                           checkpoint_callback])

# Epoch 5/5
# 72s 303ms/step - loss: 1.5599 - accuracy: 0.6182 - val_loss: 1.8425 - val_accuracy: 0.5278
```

```python
# evaluate performance on whole dataset
pre_training_results = model.evaluate(testing_data_101_10)
print(pre_training_results)

# [1.584436297416687, 0.5826930403709412]
```

### Fine-tuning the Model

```python
# unfreeze entire model
base_model.trainable = True

# keep only the last 5 layers trainable
for layer in base_model.layers[:-5]:
    layer.trainable = False
```

```python
# list all layers in model
for layer in model.layers:
    print(layer, layer.trainable)
    
# <keras.engine.input_layer.InputLayer object at 0x7f695c39b370> True
# <keras.engine.sequential.Sequential object at 0x7f695c39a260> True
# <keras.engine.functional.Functional object at 0x7f6943f0d300> True
# <keras.layers.pooling.global_average_pooling2d.GlobalAveragePooling2D object at 0x7f6943f4e830> True
# <keras.layers.core.dense.Dense object at 0x7f69400c0520> True
    
# layer 2 is the now only partly unfrozen imported model (efficientnetb0)
for layer_number, layer in enumerate(model.layers[2].layers):
    print(layer_number, layer.name, layer.trainable)
    
# 0 input_1 False
# 1 rescaling False
# 2 normalization False
# ...
# 262 block6h_se_excite False
# 263 block6h_project_conv False
# 264 block6h_project_bn False
# 265 block6h_drop True
# 266 block6h_add True
# 267 top_conv True
# 268 top_bn True
# 269 top_activation True
```

```python
# recompile the model with the new basemodel
### to prevent overfitting / to better hold on to pre-training
### the learning rate during fine-tuning should be lowered 10x
### default Adam(lr)=1e-3 => 1e-4
model.compile(loss='categorical_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
               metrics=['accuracy'])
```

```python
# continue training
fine_tuning_epochs = pretraining_epochs + 5

history_fine_tuning_model = model.fit(
                training_data_101_10,
                epochs=fine_tuning_epochs,
                # start from last pre-training checkpoint
                # training from epoch 6 - 10
                initial_epoch = history_model.epoch[-1],
                steps_per_epoch=len(training_data_101_10),
                validation_data=testing_data_101_10,
                # evaluate performance on 15% of the testing dataset
                validation_steps=int(0.15 * len(testing_data_101_10)),
                callbacks=[tensorboard_callback,
                           checkpoint_callback])

# Epoch 10/10
# 68s 287ms/step - loss: 1.0435 - accuracy: 0.7269 - val_loss: 1.6981 - val_accuracy: 0.5596
```

```python
# evaluate performance on whole dataset
fine_tuning_results = model.evaluate(testing_data_101_10)
print(fine_tuning_results)

# pre_training_results
# [1.584436297416687, 0.5826930403709412]

# fine_tuning_results
# [1.4517788887023926, 0.6104554533958435]
```

```python
# print accuracy curves
plot_accuracy_curves(history_model, "Pre-Training", history_fine_tuning_model, "Fine-Tuning")
```

![Transfer Learning](../assets/04_Tensorflow_Transfer_Learning_13.png)

```python
# the validation accuracy increase keeps slowing while training
# accuracy goes up this points to an overfitting problem
combine_training_curves(history_model, history_fine_tuning_model, pretraining_epochs=5)
```

![Transfer Learning](../assets/04_Tensorflow_Transfer_Learning_14.png)

```python
009 Saving and loading our trained model.mp4
```
