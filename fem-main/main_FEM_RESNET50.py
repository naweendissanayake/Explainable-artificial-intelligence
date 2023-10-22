#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import functools
import keract
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Configurable parameters
image_path = 'data/black_bear/ILSVRC2012_val_00045891.JPEG'


last_conv_layer_name = "conv5_block3_out"  # ResNet-50's last convolutional layer name
model_builder = tf.keras.applications.ResNet50  # Use ResNet-50 model
img_size = (224, 224)  # Change input image size to (224, 224)
preprocess_input = tf.keras.applications.resnet50.preprocess_input  # Use ResNet-50's preprocessing function
decode_predictions = tf.keras.applications.resnet50.decode_predictions  # Use ResNet-50's decoding function



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

