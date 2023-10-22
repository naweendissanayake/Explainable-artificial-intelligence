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


# Utility functions
def get_img_array(image_path, size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def expand_flat_values_to_activation_shape(values, W_layer, H_layer):
    expanded = values.reshape((1, 1, -1)) * np.ones((W_layer, H_layer, len(values)))
    return expanded

# FEM algorithm functions
def compute_binary_maps(feature_map, sigma=None):
    batch_size, W_layer, H_layer, N_channels = feature_map.shape
    thresholded_tensor = np.zeros((batch_size, W_layer, H_layer, N_channels))
    if sigma is None:
        feature_sigma = 2
    else:
        feature_sigma = sigma
    for B in range(batch_size):
        activation = feature_map[B, :, :, :]
        mean_activation_per_channel = activation.mean(axis=(0, 1))
        std_activation_per_channel = activation.std(axis=(0, 1))
        mean_activation_expanded = expand_flat_values_to_activation_shape(mean_activation_per_channel, W_layer, H_layer)
        std_activation_expanded = expand_flat_values_to_activation_shape(std_activation_per_channel, W_layer, H_layer)
        thresholded_tensor[B, :, :, :] = 1.0 * (activation > (mean_activation_expanded + feature_sigma * std_activation_expanded))
    return thresholded_tensor

def aggregate_binary_maps(binary_feature_map, orginal_feature_map):
    batch_size, W_layer, H_layer, N_channels = orginal_feature_map.shape
    orginal_feature_map = orginal_feature_map[0]
    binary_feature_map = binary_feature_map[0]
    channel_weights = np.mean(orginal_feature_map, axis=(0, 1))
    expanded_weights = expand_flat_values_to_activation_shape(channel_weights, W_layer, H_layer)
    expanded_feat_map = np.multiply(expanded_weights, binary_feature_map)
    feat_map = np.sum(expanded_feat_map, axis=2)
    if np.max(feat_map) == 0:
        return feat_map
    feat_map = feat_map / np.max(feat_map)
    return feat_map

def compute_fem(img_array, model, last_conv_layer_name):
    activations = keract.get_activations(model, img_array, auto_compile=True)
    for (k, v) in activations.items():
        if k == last_conv_layer_name:
            feature_map = v
    binary_feature_map = compute_binary_maps(feature_map)
    saliency = aggregate_binary_maps(binary_feature_map, feature_map)
    return saliency

def save_fem_superimposed_visualization(image_path, saliency, superimposed_img_path=image_path, alpha=0.4):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * saliency)
    turbo = cm.get_cmap("turbo")
    turbo_colors = turbo(np.arange(256))[:, :3]
    turbo_heatmap = turbo_colors[heatmap]
    turbo_heatmap = tf.keras.preprocessing.image.array_to_img(turbo_heatmap)
    turbo_heatmap = turbo_heatmap.resize((img.shape[1], img.shape[0]))
    turbo_heatmap = tf.keras.preprocessing.image.img_to_array(turbo_heatmap)
    superimposed_img = turbo_heatmap * alpha + img * 0.2
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(superimposed_img_path)
    return superimposed_img_path

