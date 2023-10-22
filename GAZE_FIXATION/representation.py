#!/usr/bin/env python
# coding: utf-8

# # SaliencyMap Class:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class SaliencyMap:
    def __init__(self, saliency_path):
        self.saliency = Image.open(saliency_path)
        self.saliency_array = np.array(self.saliency)

    def represent_heatmap(self, cmap):
        # Calculate the minimum and maximum values in the saliency map
        min_val = np.min(self.saliency_array)
        max_val = np.max(self.saliency_array)

        # Normalize the saliency map to the range [0, 1]
        normalized_saliency_map = (self.saliency_array - min_val) / (max_val - min_val)

        # Create a subplot with two columns
        plt.figure(figsize=(12, 3))

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.saliency_array, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        # Plot the normalized heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(normalized_saliency_map, cmap=cmap)
        plt.title('Normalized Heatmap')
        plt.axis('off')

        plt.colorbar()

        plt.show()

    def represent_heatmap_overlaid(self, image_path, cmap):
        overlay_image = Image.open(image_path)

        # Calculate the minimum and maximum values in the saliency map
        min_val = np.min(self.saliency_array)
        max_val = np.max(self.saliency_array)

        # Normalize the saliency map to the range [0, 1]
        normalized_saliency_map = (self.saliency_array - min_val) / (max_val - min_val)
        normalized_saliency_map = plt.cm.gist_heat(normalized_saliency_map)

        overlay2 = Image.blend(overlay_image.convert("RGBA"),
            Image.fromarray((normalized_saliency_map * 255).astype(np.uint8)).convert("RGBA"), alpha=0.7)

        plt.figure(figsize=(12, 3))

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(overlay_image)
        plt.title('Original Image')
        plt.axis('off')

        # Plot the overlaid heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(overlay2)
        plt.title('Overlaid Heatmap')
        plt.axis('off')
        plt.colorbar()
        plt.show()

    def represent_isolines(self, cmap):
        empty_ = np.zeros(self.saliency_array.shape)

        # Calculate the minimum and maximum values in the saliency map
        min_val = np.min(self.saliency_array)
        max_val = np.max(self.saliency_array)

        # Create isoline representation with the specified colormap for signed maps
        num_contours = 15
        contour_levels = np.linspace(min_val, max_val, num_contours)
        plt.figure(figsize=(empty_.shape[1] / 100, empty_.shape[0] / 100))  # Set figure size
        contour_plot = plt.contour(self.saliency_array, levels=contour_levels, cmap=cmap)
        isoline_image = plt.imshow(empty_, cmap=cmap)

        plt.axis('off')

        return isoline_image

    def represent_isolines_superimposed(self, image_path, cmap):
        overlay_image = Image.open(image_path)

        # Calculate the minimum and maximum values in the saliency map
        min_val = np.min(self.saliency_array)
        max_val = np.max(self.saliency_array)

        # Create isoline representation with the specified colormap for signed maps
        num_contours = 15
        contour_levels = np.linspace(min_val, max_val, num_contours)
        plt.figure(figsize=(self.saliency_array.shape[1] / 300, self.saliency_array.shape[0] / 300))  # Set figure size
        contour_plot = plt.contour(self.saliency_array, levels=contour_levels, cmap=cmap)
        isoline_image = np.zeros_like(self.saliency_array)  # Initialize an empty array

        # Convert contour lines to an image
        for path in contour_plot.collections:
            for line in path.get_paths():
                for segment in line.iter_segments():
                    isoline_image[int(segment[0][1]), int(segment[0][0])] = 1

        # Overlay isolines on the original image
        overlay2 = Image.blend(overlay_image.convert("RGBA"),
                               Image.fromarray((isoline_image * 255).astype(np.uint8)).convert("RGBA"),
                               alpha=0.7)

        plt.figure(figsize=(12, 3))

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(overlay_image)
        plt.title('Original Image')
        plt.axis('off')

        # Plot the overlaid isolines
        plt.subplot(1, 2, 2)
        plt.imshow(overlay2)
        plt.title('Overlaid Isolines')
        plt.axis('off')

        plt.show()

        return overlay2

    def represent_hard_selection(self, image_path, threshold):
        overlay_image = Image.open(image_path)
        image_array = np.array(overlay_image)

        # Create a mask where saliency values meet the threshold
        mask = self.saliency_array >= threshold

        # Create an empty image
        result_image = np.zeros_like(image_array)

        # Copy pixels from the original image where the mask is True
        result_image[mask] = image_array[mask]

        plt.figure(figsize=(12, 3))

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(overlay_image)
        plt.title('Original Image')
        plt.axis('off')

        # Plot the overlaid heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(result_image)
        plt.title('Overlaid Heatmap')
        plt.axis('off')
        plt.colorbar()
        plt.show()

        return result_image

    def represent_soft_selection(self, image_path, threshold):
        overlay_image = Image.open(image_path)
        image_array = np.array(overlay_image)

        # Normalize the saliency map to the range [0, 1]
        normalized_saliency = self.saliency_array / 255.0

        #  Soft(x, y) = Img(x, y) * Saliency(x, y)
        soft_selection = (image_array.astype(float) * normalized_saliency[:, :, np.newaxis]).astype(np.uint8)

        # Create an Image object from the resulting array
        soft_selection_image = Image.fromarray(soft_selection)

        plt.figure(figsize=(12, 3))
              
        plt.imshow(overlay_image)
        plt.title('Original Image')
        plt.axis('off')

        # Plot the overlaid heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(soft_selection_image)
        plt.title('Overlaid Heatmap')
        plt.axis('off')
        plt.colorbar()
        plt.show()
        #extra work 
    def compare_side_by_side(self, other_saliency_map, title1, title2):
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        fig.suptitle('Saliency Map Comparison')
        axes[0].imshow(self.saliency_array, cmap='gist_heat')
        axes[0].set_title(title1)
        axes[0].axis('off')
        axes[1].imshow(other_saliency_map.saliency_array, cmap='gist_heat')
        axes[1].set_title(title2)
        axes[1].axis('off')
        plt.show()

