import os
import random
import time
import math
from collections import OrderedDict
import numpy as np
import logging

import torch

from os2d.data.voc_eval import do_voc_evaluation

from os2d.utils import add_to_meters_in_dict, print_meters, time_since, time_for_printing
from os2d.modeling.model import get_feature_map_size_for_network
from os2d.structures.feature_map import FeatureMapSize

import os2d.utils.visualization as visualizer


@torch.no_grad() # do evaluation in forward mode (for speed and memory)
def evaluate(dataloader, net, cfg, criterion=None, print_per_class_results=False):
    """
    在一个数据集上评估所提供的模型

    Args:
        dataloader - the dataloader to get data
        net - the network to use
        cfg - config with all the parameters
        criterion - criterion (usually the same one as used for training), can be None, will just not compute related metrics
        print_per_class_results - flag showing whether to printout extra data (per class AP) - usually used at the final evaluation

    Returns:
        losses (OrderedDict) - all computed metrics, e.g., losses["mAP@0.50"] - mAP at IoU threshold 0.5
    """
    logger = logging.getLogger("OS2D.evaluate")
    dataset_name = dataloader.get_name()
    dataset_scale = dataloader.get_eval_scale()
    logger.info("开始在数据集{0}上进行评估, 评估数量是 {1}".format(dataset_name, dataset_scale))
    t_start_eval = time.time()
    net.eval()

    iterator = make_iterator_extract_scores_from_images_batched(dataloader, net, logger,
                                                                image_batch_size=cfg.eval.batch_size,
                                                                is_cuda=cfg.is_cuda,
                                                                class_image_augmentation=cfg.eval.class_image_augmentation)

    boxes = []
    gt_boxes = []
    losses = OrderedDict()
    image_ids = []

    # loop over all dataset images
    num_evaluted_images = 0
    eval_idx = 0
    for data in iterator:
        logger.info(f"评估数据的第: {eval_idx} 条")
        eval_idx += 1
        image_id, image_loc_scores_pyramid, image_class_scores_pyramid,\
                    image_pyramid, query_img_sizes, class_ids,\
                    box_reverse_transform, image_fm_sizes_p, transform_corners_pyramid\
                    = data
        image_ids.append(image_id)
        # img_size_pyramid: [FeatureMapSize(w=1280, h=960)]
        num_evaluted_images += 1
        img_size_pyramid = [FeatureMapSize(img=img) for img in image_pyramid]
        # num_labels： 类别数量,185
        num_labels = len(class_ids)
        # gt_boxes_one_image: 获取image_id对应的图片的bbox信息，真实的bbox信息，ground truth
        gt_boxes_one_image = dataloader.get_image_annotation_for_imageid(image_id)
        gt_boxes.append(gt_boxes_one_image)
        # compute losses
        if len(gt_boxes_one_image) > 0:
            # 这张图片标注信息，gt_labels_one_image获取ground truth
            gt_labels_one_image = gt_boxes_one_image.get_field("labels")
            # gt_boxes_one_image中的全局标签类别已经被换成本地的标签类别
            dataloader.update_box_labels_to_local(gt_boxes_one_image, class_ids)

            loc_targets_pyramid, class_targets_pyramid = \
                    dataloader.box_coder.encode_pyramid(gt_boxes_one_image,
                                                        img_size_pyramid, num_labels,
                                                        default_box_transform_pyramid=box_reverse_transform)

            # 返回原始标签
            gt_boxes_one_image.add_field("labels", gt_labels_one_image)

            # 可视化 GT 以进行调试
            if cfg.visualization.eval.show_gt_boxes:
                visualizer.show_gt_boxes(image_id, gt_boxes_one_image, class_ids, dataloader)

            if cfg.is_cuda:
                loc_targets_pyramid = [loc_targets.cuda() for loc_targets in loc_targets_pyramid]
                class_targets_pyramid = [class_targets.cuda() for class_targets in class_targets_pyramid]
                transform_corners_pyramid = [transform_corners.cuda() for transform_corners in transform_corners_pyramid]

            add_batch_dim = lambda list_of_tensors: [t.unsqueeze(0) for t in list_of_tensors]
            if criterion is not None:
                # 如果提供了criterion，则使用它来计算它可以计算的所有指标
                losses_iter = criterion(add_batch_dim(image_loc_scores_pyramid) if image_loc_scores_pyramid[0] is not None else None,
                                        add_batch_dim(loc_targets_pyramid),
                                        add_batch_dim(image_class_scores_pyramid),
                                        add_batch_dim(class_targets_pyramid)
                                        )
            
                # convert to floats
                for l in losses_iter:
                    losses_iter[l] = losses_iter[l].mean().item()
                # printing
                print_meters(losses_iter, logger)
                # update logs
                add_to_meters_in_dict(losses_iter, losses)
        
        # decode image predictions
        boxes_one_image = \
            dataloader.box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                                img_size_pyramid, class_ids,
                                                nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                                nms_score_threshold=cfg.eval.nms_score_threshold,
                                                inverse_box_transforms=box_reverse_transform,
                                                transform_corners_pyramid=transform_corners_pyramid)

        boxes.append(boxes_one_image.cpu())
        # 显示从查询图检测到的目标bbox
        if cfg.visualization.eval.show_detections:
            visualizer.show_detection_from_dataloader(boxes_one_image, image_id, dataloader, cfg.visualization.eval, class_ids=None)
        
        if cfg.visualization.eval.show_class_heatmaps:
            visualizer.show_class_heatmaps(image_id, class_ids, image_fm_sizes_p, class_targets_pyramid, image_class_scores_pyramid,
                                            cfg_local=cfg.visualization.eval,
                                            class_image_augmentation=cfg.eval.class_image_augmentation)

        if cfg.is_cuda:
            torch.cuda.empty_cache()

    # normalize by number of steps
    for k in losses:
        losses[k] /= num_evaluted_images

    # Save detection if requested
    path_to_save_detections = cfg.visualization.eval.path_to_save_detections
    if path_to_save_detections:
        data = {"image_ids" : image_ids,
                "boxes_xyxy" : [bb.bbox_xyxy for bb in boxes], 
                "labels" : [bb.get_field("labels") for bb in boxes],
                "scores" : [bb.get_field("scores") for bb in boxes],
                "gt_boxes_xyxy" : [bb.bbox_xyxy for bb in gt_boxes],
                "gt_labels" : [bb.get_field("labels") for bb in gt_boxes],
                "gt_difficults" : [bb.get_field("difficult") for bb in gt_boxes]
        }
        dataset_name = dataloader.get_name()
        os.makedirs(path_to_save_detections, exist_ok=True)
        save_path = os.path.join(path_to_save_detections, dataset_name + "_detections.pth")
        torch.save(data, save_path) 

    # compute mAP
    for mAP_iou_threshold in cfg.eval.mAP_iou_thresholds:
        logger.info("Evaluating at IoU th {:0.2f}".format(mAP_iou_threshold))
        ap_data = do_voc_evaluation(boxes, gt_boxes, iou_thresh=mAP_iou_threshold, use_07_metric=False)
        losses["mAP@{:0.2f}".format(mAP_iou_threshold)] = ap_data["map"]
        losses["mAPw@{:0.2f}".format(mAP_iou_threshold)] = ap_data["map_weighted"]
        losses["recall@{:0.2f}".format(mAP_iou_threshold)] = ap_data["recall"]
        losses["AP_joint_classes@{:0.2f}".format(mAP_iou_threshold)] = ap_data["ap_joint_classes"]

        if print_per_class_results:
            # per class AP information
            for i_class, (ap, recall, n_pos) in enumerate(zip(ap_data["ap_per_class"], ap_data["recall_per_class"], ap_data["n_pos"])):
                if not np.isnan(ap):
                    assert i_class in class_ids, "Could not find class_id in the list of ids"
                    logger.info("Class {0} (local {3}), AP {1:0.4f}, #obj {2}, recall {4:0.4f}".format(i_class,
                                                                                                       ap,
                                                                                                       n_pos,
                                                                                                       class_ids.index(i_class),
                                                                                                       recall))
    # save timing
    losses["eval_time"] = (time.time() - t_start_eval)
    logger.info("数据集 {0} 评估完成, scale {1}".format(dataset_name, dataset_scale))
    print_meters(losses, logger)
    return losses


def make_iterator_extract_scores_from_images_batched(dataloader, net, logger, image_batch_size, is_cuda,
                                                     num_random_pyramid_scales=0, num_random_negative_labels=-1,
                                                     class_image_augmentation=""):
    """
    生成器循环遍历数据集并将模型应用于所有元素。
    迭代器将一张一张地遍历图像。
    用于 evaluate 和 .train.mine_hard_patches

    Args:
        dataloader - 数据加载器获取数据
        net - the network to use
        logger - the created logger
        image_batch_size (int) - 一批中要放入的图像数量
        is_cuda (bool) - use GPUs or not
        num_random_pyramid_scales (int) - 要尝试的随机金字塔比例的数量，默认值 (0) 表示来自配置的标准比例
            passed to dataloader.make_iterator_for_all_images
        num_random_negative_labels (int) - 要尝试的随机负标签数，默认 (-1) 表示添加所有可能的标签
        class_image_augmentation (str) - 要进行的类图像增强类型, default - no augmentation, support "rotation90" and "horflip"

    Returns:
        在数据元组上创建一个迭代器：
        image_id (int)
        image_loc_scores_p (list of tensors) - 在解码时获得bounding boxes的定位分数
            len(image_loc_scores_p) = num pyramid levels, tensor size: num_labels x 4 x num_anchors
        image_class_scores_p (list of tensors) - clasification scores to recognize classes when decoding
            len(image_class_scores_p) = num pyramid levels, tensor size: num_labels x num_anchors
        one_image_pyramid (list of tensors) - input images at all pyramid levels
        batch_query_img_sizes (list of FeatureMapSize) - sizes of used query images (used in mine_hard_patches)
            len(batch_query_img_sizes) = num query images
        batch_class_ids (list of int) - class ids of used query images; len(batch_class_ids) = num query images,
        box_reverse_transforms (list of os2d.structures.transforms.TransformList) - reverse transforms to convert boxes
            from the coordinates of each resized image to the original global coordinates
            len(box_reverse_transforms) = num pyramid levels
        image_fm_sizes_p (list of FeatureMapSize) - sizes of the feature maps of the current pyramid
            len(image_fm_sizes_p) = num pyramid levels
        transform_corners_p (list of tensors) - corners of the parallelogram after the transformation mapping (used for visualization)
            len(transform_corners_p) = num pyramid levels, tensor size: num_labels x 8 x num_anchors
    """

    logger.info("从所有图像中提取分数")
    # 获取所有类的图像, class_images: list(tensor), 185,[1,3,265,216], [1,RGB,w,h] ,class_aspect_ratios:每个类的特征图的大小list,(w,h), class_ids: 类别id, list
    class_images, class_aspect_ratios, class_ids = dataloader.get_all_class_images()
    num_classes = len(class_images)  #类别图像数量，eg: 28
    assert len(class_aspect_ratios) == num_classes
    assert len(class_ids) == num_classes
    query_img_sizes = [FeatureMapSize(img=img) for img in class_images]
    
    # 当前代码仅适用于 class batch == 1，这在某些地方效率低下，但在其他地方很好
    # is there a better way?
    class_batch_size = 1

    # 从批次的类图像中提取所有类卷积
    class_conv_layer_batched = []
    logger.info("从 {0} 个类别中提取权重。{1}".format(num_classes,
        f" with {class_image_augmentation} augmentation" if class_image_augmentation else ""))
    for i in range(0, num_classes, class_batch_size):
        batch_class_ids = class_ids[i : i + class_batch_size]

        batch_class_images = []
        for i_label in range(len(batch_class_ids)):
            im = class_images[i + i_label].squeeze(0)
            if is_cuda:
                im = im.cuda()
            batch_class_images.append(im)  # eg: [3,240,240]
            if not class_image_augmentation:
                num_class_views = 1
            elif class_image_augmentation == "rotation90":
                im90 = im.rot90(1, [1, 2])
                im180 = im90.rot90(1, [1, 2])
                im270 = im180.rot90(1, [1, 2])
                batch_class_images.append(im90)
                batch_class_images.append(im180)
                batch_class_images.append(im270)
                num_class_views = 4
            elif class_image_augmentation == "horflip":
                im_flipped = im.flip(2)
                batch_class_images.append(im_flipped)
                num_class_views = 2
            elif class_image_augmentation == "horflip_rotation90":
                im90 = im.rot90(1, [1, 2])
                im180 = im90.rot90(1, [1, 2])
                im270 = im180.rot90(1, [1, 2])
                im_flipped = im.flip(2)
                im90_flipped = im90.flip(2)
                im180_flipped = im180.flip(2)
                im270_flipped = im270.flip(2)

                for new_im in [im90, im180, im270, im_flipped, im90_flipped, im180_flipped, im270_flipped]:
                    batch_class_images.append(new_im)

                num_class_views = len(batch_class_images)
            else:
                raise RuntimeError(f"Unknown value of class_image_augmentation: {class_image_augmentation}")

        for b_im in batch_class_images:
            class_feature_maps = net.net_label_features([b_im])   #前向传播，提取图片的类别特征图
            # Os2Head实例
            class_conv_layer = net.os2d_head_creator.create_os2d_head(class_feature_maps)
            # 加到一个列表中
            class_conv_layer_batched.append(class_conv_layer)
    
    # 遍历所有图像， image_batch_size：1, num_random_pyramid_scales:0
    iterator_batches = dataloader.make_iterator_for_all_images(image_batch_size, num_random_pyramid_scales=num_random_pyramid_scales)
    for batch_ids, pyramids_batch, box_transforms_batch, initial_img_size_batch in iterator_batches:
        # batch_ids: 批次id, [6], pyramids_batch: list，这个批次的特征金字塔, box_transforms_batch:bbox框信息，initial_img_size_batch:list，初始特征图尺寸 [FeatureMapSize(w=3264, h=2448)]
        t_start_batch = time.time()
        # 选择要用于此批次搜索的标签
        if num_random_negative_labels >= 0 :
            neg_labels = torch.randperm(len(class_conv_layer_batched)) #随机打乱标签
            neg_labels = neg_labels[:num_random_negative_labels]
            # add positive labels
            pos_labels = dataloader.get_class_ids_for_image_ids(batch_ids)
            pos_labels = dataloader.convert_label_ids_global_to_local(pos_labels, class_ids)
            batch_labels_local = torch.cat([neg_labels, pos_labels], 0).unique()
        else:
            # take all the labels - needed for evaluation， batch_labels_local：list, [185]
            batch_labels_local = torch.arange(len(class_conv_layer_batched))
        # 这个批次中的每个类别的id, batch_class_ids: [185], 真实的标签id
        batch_class_ids = [class_ids[l // num_class_views] for l in batch_labels_local]
        # 批次中所有查询图像的特征图尺寸
        batch_query_img_sizes = [query_img_sizes[l // num_class_views] for l in batch_labels_local]
        # 提取所有金字塔级别的特征
        batch_images_pyramid = []
        loc_scores = []
        class_scores = []
        fm_sizes = []
        transform_corners = []
        num_pyramid_levels = len(pyramids_batch)
        
        t_cum_features = 0.0
        t_cum_labels = 0.0
        for batch_images in pyramids_batch:
            if is_cuda:
                batch_images = batch_images.cuda()
            # batch_images： [1,3,960,1280], 原始图片特征， 这里是每张图片，1代表1张图片
            t_start_features = time.time()
            feature_maps = net.net_feature_maps(batch_images)   #resnet提取后的特征, [1,1024,60,80]，每个特征金字塔
            torch.cuda.synchronize()
            t_cum_features += time.time() - t_start_features

            # batch class images
            loc_scores.append([])
            class_scores.append([])
            fm_sizes.append([])
            transform_corners.append([])
            t_start_labels = time.time()
            assert class_batch_size == 1, "the iterator on images works only with labels batches of size 1"

            for i_class_batch in batch_labels_local:
                #在此金字塔级别应用网络模型
                loc_s_p, class_s_p, _, fm_sizes_p, transform_corners_p = \
                     net(class_head=class_conv_layer_batched[i_class_batch],
                         feature_maps=feature_maps)
                loc_scores[-1].append(loc_s_p)
                class_scores[-1].append(class_s_p)
                fm_sizes[-1].append(fm_sizes_p)
                transform_corners[-1].append(transform_corners_p)
            torch.cuda.synchronize()
            t_cum_labels += time.time() - t_start_labels

            if not feature_maps.requires_grad:
                # explicitly remove a possibly large chunk of GPU memory
                del feature_maps

            batch_images_pyramid.append(batch_images)

        timing_str = "提取特征耗时: {0}, 标签判断耗时: {1}, ".format(time_for_printing(t_cum_features, mode="s"),
                                                          time_for_printing(t_cum_labels, mode="s"))

        # loc_scores, class_scores: pyramid_level x class_batch x image_in_batch x
        for i_image_in_batch, image_id in enumerate(batch_ids):
            # 从所有金字塔级别获得分数
            image_loc_scores_p, image_class_scores_p, image_fm_sizes_p = [], [], []
            transform_corners_p = []
            for i_p in range(num_pyramid_levels):
                if loc_scores is not None and loc_scores[0] is not None and loc_scores[0][0] is not None:
                    image_loc_scores_p.append(torch.cat([s[i_image_in_batch] for s in loc_scores[i_p]], 0))
                else:
                    image_loc_scores_p.append(None)
                image_class_scores_p.append(torch.cat([s[i_image_in_batch] for s in class_scores[i_p]], 0))
                # transform_corners_p list: 元素: [185,8,4800]
                if transform_corners is not None and transform_corners[0] is not None and transform_corners[0][0] is not None:
                    transform_corners_p.append(torch.cat([s[i_image_in_batch] for s in transform_corners[i_p]], 0))
                else:
                    transform_corners_p.append(None)

                image_fm_sizes_p.append(fm_sizes[i_p][0])

            # 得到一个图像的金字塔  [i_p], list, eg: 7* [3,960,1280], 7种类型特征金字塔
            one_image_pyramid = [p[i_image_in_batch] for p in batch_images_pyramid]

            # 提取box变换
            box_reverse_transforms = box_transforms_batch[i_image_in_batch]

            logger.info(timing_str + "整个网络耗时: {0}".format(time_since(t_start_batch)))
            # image_id: 6, image_loc_scores_p:list, 7([185,4,1200|1900|3072|...|12288]),  这张图片的id
            # image_class_scores_p:list, 7*(185,1200|1900|3072|...|12288])   #7个特征金字塔中，bbox的分类分数
            # one_image_pyramid: list, 7*[(3,480,640),(3,600,800),...(3,1536,2048) # 7个特征金字塔的大小
            # batch_query_img_sizes: 185 eg: (FeatureMapSize(w=216, h=265)...FeatureMapSize(w=178, h=322)) 每个类别的特征图尺寸
            # batch_class_ids: list 185, 每个列表的id
            # box_reverse_transforms: 7* (每个特征金字塔的感受野）
            # image_fm_sizes_p: 每个特征金字塔特征图大小
            # transform_corners_p： 7* ([185,8,1200|1900|3072|...|12288])
            yield image_id, image_loc_scores_p, image_class_scores_p, one_image_pyramid,\
                  batch_query_img_sizes, batch_class_ids, box_reverse_transforms, image_fm_sizes_p, transform_corners_p
