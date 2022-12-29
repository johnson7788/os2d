#!/usr/bin/env python
# coding: utf-8

# This is a demo illustrating an application of the OS2D method on one image.
# Demo assumes the OS2D code is [installed](./INSTALL.md).

# In[1]:


import os
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from os2d.modeling.model import build_os2d_from_config
from os2d.config import cfg
import  os2d.utils.visualization as visualizer
from os2d.structures.feature_map import FeatureMapSize
from os2d.utils import setup_logger, read_image, get_image_size_after_resize_preserving_aspect_ratio

logger = setup_logger("OS2D")


# In[2]:


# use GPU if have available
cfg.is_cuda = torch.cuda.is_available()


# Download the trained model (is the script does not work download from [Google Drive](https://drive.google.com/open?id=1l_aanrxHj14d_QkCpein8wFmainNAzo8) and put to models/os2d_v2-train.pth). See [README](./README.md) to get links for other released models.

# In[3]:


# In[4]:


cfg.init.model = "models/os2d_v2-train.pth"
net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)


# Get the image where to detect and two class images.

# In[5]:


input_image = read_image("data/demo/input_image.jpg")
class_images = [read_image("data/demo/class_image_0.jpg"),
                read_image("data/demo/class_image_1.jpg")]
class_ids = [0, 1]


# 使用 torchvision 将图像转换为 torch.Tensor 并应用归一化。


transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])


#准备输入图像


h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                               w=input_image.size[0],
                                                               target_size=1500)
input_image = input_image.resize((w, h))

input_image_th = transform_image(input_image)
input_image_th = input_image_th.unsqueeze(0)
if cfg.is_cuda:
    input_image_th = input_image_th.cuda()


# 准备类别图片

class_images_th = []
for class_image in class_images:
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
                                                               w=class_image.size[0],
                                                               target_size=cfg.model.class_image_size)
    class_image = class_image.resize((w, h))

    class_image_th = transform_image(class_image)
    if cfg.is_cuda:
        class_image_th = class_image_th.cuda()

    class_images_th.append(class_image_th)


# Run the network with one command


with torch.no_grad():
     loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_image_th, class_images=class_images_th)

# 或者，可以单独运行模型的各个阶段，这很方便，例如，用于在许多输入图像之间共享类特征提取。
# with torch.no_grad():
#     feature_map = net.net_feature_maps(input_image_th)

#     class_feature_maps = net.net_label_features(class_images_th)
#     class_head = net.os2d_head_creator.create_os2d_head(class_feature_maps)

#     loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(class_head=class_head,
#                                                                                             feature_maps=feature_map)

# 将按批次组织的图像转换为按金字塔级别组织的图像。演示中不需要，但对于批次中的多个图像和多个金字塔级别必不可少。


image_loc_scores_pyramid = [loc_prediction_batch[0]]
image_class_scores_pyramid = [class_prediction_batch[0]]
img_size_pyramid = [FeatureMapSize(img=input_image_th)]
transform_corners_pyramid = [transform_corners_batch[0]]


# 将网络输出解码为检测框

boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                           img_size_pyramid, class_ids,
                                           nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                           nms_score_threshold=cfg.eval.nms_score_threshold,
                                           transform_corners_pyramid=transform_corners_pyramid)

# 删除一些字段以减轻可视化
boxes.remove_field("default_boxes")

# 请注意，系统输出 [-1, 1] 段中的相关作为检测分数（越高检测越好）。
scores = boxes.get_field("scores")


# 显示类别图片

figsize = (8, 8)
fig=plt.figure(figsize=figsize)
columns = len(class_images)
for i, class_image in enumerate(class_images):
    fig.add_subplot(1, columns, i + 1)
    plt.imshow(class_image)
    plt.axis('off')

# 显示固定数量的高于特定阈值的检测。黄色矩形显示检测框。每个box子都有一个类别标签和检测分数（越高的检测效果越好）。红色平行四边形说明了将类图像与检测位置的输入图像对齐的仿射变换。

plt.rcParams["figure.figsize"] = figsize

cfg.visualization.eval.max_detections = 8
cfg.visualization.eval.score_threshold = float("-inf")
visualizer.show_detections(boxes, input_image,
                           cfg.visualization.eval)

