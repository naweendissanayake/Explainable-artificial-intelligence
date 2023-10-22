import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import cv2

def load_and_preprocess_image(image_path, target_size):
    image = np.array(load_img(image_path, target_size=target_size))
    return image

def get_model_and_layers():
    model = ResNet50()
    last_conv_layer = model.get_layer("conv5_block3_out")
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ["avg_pool", "predictions"]:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    return model, last_conv_layer_model, classifier_model, last_conv_layer

def calculate_gradcam(image, model, last_conv_layer_model, classifier_model, last_conv_layer):
    with tf.GradientTape() as tape:
        inputs = image[np.newaxis, ...]
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    gradcam = np.mean(last_conv_layer_output, axis=-1)
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam) # relu
    gradcam = cv2.resize(gradcam, (224, 224))

    return gradcam

def calculate_guided_gradcam(image, last_conv_layer, last_conv_layer_model):
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(image[np.newaxis, ...])
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)[0]
    last_conv_layer_output = last_conv_layer_output[0]

    guided_grads = (
        tf.cast(last_conv_layer_output > 0, "float32")
        * tf.cast(grads > 0, "float32")
        * grads
    )

    pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))
    guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)

    for i, w in enumerate(pooled_guided_grads):
        guided_gradcam += w * last_conv_layer_output[:, :, i]

    guided_gradcam = cv2.resize(guided_gradcam.numpy(), (224, 224))
    guided_gradcam = np.clip(guided_gradcam, 0, np.max(guided_gradcam))
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / (guided_gradcam.max() - guided_gradcam.min())

    return guided_gradcam

