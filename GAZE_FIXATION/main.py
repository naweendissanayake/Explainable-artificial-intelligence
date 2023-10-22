#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from representation import SaliencyMap


saliency_path = 'test_saliency_img.png'
image_path = 'test.jpg'
threshold=190

# image_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_N_1.png'
# saliency_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_GFDM_N_1.png'
# threshold=50

#image_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_N_2.png'
#saliency_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_GFDM_N_2.png'
#threshold=100

saliency_map = SaliencyMap(saliency_path)

saliency_map.represent_heatmap(cmap='gist_heat')
saliency_map.represent_heatmap_overlaid(image_path, cmap='gist_heat')
saliency_map.represent_isolines(cmap='gist_heat')
saliency_map.represent_isolines_superimposed(image_path, cmap='gist_heat')
saliency_map.represent_hard_selection(image_path, threshold=threshold)
saliency_map.represent_soft_selection(image_path, threshold=threshold)


#extra
pairs = [
    {
        'image_path': '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_N_1.png',
        'saliency_map_path': '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_GFDM_N_1.png',
        'title': 'Pair 1'
    },
    {
        'image_path': 'test.jpg',
        'saliency_map_path': 'test_saliency_img.png',
        'title': 'Pair 2'
    },
    {
         'image_path': '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_N_2.png',
         'saliency_map_path': '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps/Colonial_AcademiaDeBellasArtes_Queretaro_GFDM_N_2.png',
        'title': 'Pair 3'
    }
]

for pair in pairs:
    saliency_map = SaliencyMap(pair['saliency_map_path'])

    # Visualize different representations
    saliency_map.represent_heatmap('gist_heat')
    saliency_map.represent_heatmap_overlaid(pair['image_path'], 'gist_heat')
    saliency_map.represent_isolines('gist_heat')
    saliency_map.represent_isolines_superimposed(pair['image_path'], 'gist_heat')
    saliency_map.represent_hard_selection(pair['image_path'], threshold=190)
    saliency_map.represent_soft_selection(pair['image_path'], threshold=190)

    # Compare the saliency map with other maps
    for extra_pair in pairs:
        if extra_pair != pair:
            extra_saliency_map = SaliencyMap(extra_pair['saliency_map_path'])
            saliency_map.compare_side_by_side(
                extra_saliency_map,
                title1=pair['title'],
                title2=extra_pair['title']
            )

