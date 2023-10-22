# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from keras.preprocessing import image
import RISE_resnet50

# Create an instance of the Model class
model = Model()

# Load an image and preprocess it
image_path = 'data/black_bear/ILSVRC2012_val_00000592.JPEG'
img, x = load_img(image_path)

# Set the parameters for RISE
N = 2000
s = 8
p1 = 0.5
num_masks_to_display = 4

# Generate masks for RISE
masks = generate_masks(N, s, p1)

# Display some of the generated masks
fig, axs = plt.subplots(1, num_masks_to_display, figsize=(15, 3))
for i in range(num_masks_to_display):
    ax = axs[i]
    im = ax.imshow(masks[i, :, :, 0], cmap='jet')
    ax.set_title(f'Mask {i+1}')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
plt.tight_layout()
plt.show()

# Explain the model's prediction using RISE
sal = explain(model, x, masks)

# Overlay the saliency map on the input image
class_idx = 243  # Replace with the index of the class you want to explain
plt.title('Explanation for `{}`'.format(class_name(class_idx)))
plt.axis('off')
plt.imshow(img)
plt.imshow(sal[class_idx], cmap='jet', alpha=0.5)
plt.show()


%run /path/to/RISE_resnet50.py
