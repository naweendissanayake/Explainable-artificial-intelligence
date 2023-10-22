#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import tensorflow as tf
import functools
import keract

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

image_path = 'data/black_bear/ILSVRC2012_val_00045891.JPEG'
# last_conv_layer_name = "block5_conv3"
# classifier_layer_names = ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
last_conv_layer_name = "block5_conv3"  # VGG16's last convolutional layer name
model_builder = tf.keras.applications.VGG16  # Use VGG16 model
img_size = (224, 224)  # Change input image size to (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input  # Use VGG16's preprocessing function
decode_predictions = tf.keras.applications.vgg16.decode_predictions  # Use VGG16's decoding function

# Prepare image
img_array = preprocess_input(get_img_array(image_path, size=img_size))

# Make model
model = model_builder(weights="imagenet")
model.layers[-1].activation = None

# Generate saliency with FEM algorithm
saliency = compute_fem(img_array, model, last_conv_layer_name)

# Print top predicted class
preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

# Display saliency as a heatmap
plt.matshow(saliency, cmap="turbo", vmin=0, vmax=1, interpolation="gaussian")
plt.show()

# Create superimposed visualization
path = save_fem_superimposed_visualization(image_path, saliency)
display(Image(path))


# In[ ]:




