import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import cm
from matplotlib import pyplot as plt

# Define constants and functions
model_builder = keras.applications.vgg16.VGG16
img_size = (224, 224)
preprocess_input = keras.applications.vgg16.preprocess_input
decode_predictions = keras.applications.vgg16.decode_predictions
last_conv_layer_name = "block5_conv3"
classifier_layer_names = ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']


# Load and preprocess the image
image_path = 'data/African_elephant/ILSVRC2012_val_00001177.JPEG'
img_array = preprocess_input(get_img_array(image_path, size=img_size))

# Create the VGG16 model
model = model_builder(weights="imagenet")
model.summary()

# Predict the image
preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate and visualize the heatmap
heatmap = make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
)
heatmap.shape

# Display the heatmap
plt.matshow(heatmap)
plt.show()

# Load and preprocess the original image
img = keras.preprocessing.image.load_img(image_path)
img = keras.preprocessing.image.img_to_array(img)

# Rescale heatmap to a range 0-255
heatmap = np.uint8(255 *heatmap)

# Use the jet colormap to colorize the heatmap
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]

# Create an image with RGB colorized heatmap
jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)


superimposed_img = jet_heatmap * 0.5 + img
superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
plt.imshow(superimposed_img)
plt.show()

