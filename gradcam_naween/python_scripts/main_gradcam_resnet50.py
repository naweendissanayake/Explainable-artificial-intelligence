import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2


image_path = 'data/black_bear/ILSVRC2012_val_00035157.JPEG'
image_size = (224, 224, 3)

image = load_and_preprocess_image(image_path, image_size)
model, last_conv_layer_model, classifier_model, last_conv_layer = get_model_and_layers()

gradcam = calculate_gradcam(image, model, last_conv_layer_model, classifier_model, last_conv_layer)
guided_gradcam = calculate_guided_gradcam(image, last_conv_layer, last_conv_layer_model)

# Visualization code
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(gradcam, cmap='viridis')
plt.title('Grad-CAM')

plt.subplot(2, 2, 3)
plt.imshow(guided_gradcam, cmap='viridis')
plt.title('Guided Grad-CAM')

plt.subplot(2, 2, 4)
plt.imshow(image)
plt.imshow(guided_gradcam, alpha=0.5)
plt.title('Overlay: Original + Guided Grad-CAM')

plt.show()

