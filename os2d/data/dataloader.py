import os
import sys
import random
import math
import copy
import logging
from collections import OrderedDict
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os2d.engine.augmentation import DataAugmentation
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList, cat_boxlist
from os2d.structures.transforms import TransformList, crop
import os2d.structures.transforms as transforms_boxes
import os2d.utils.visualization as visualizer
from .dataset import build_dataset_by_name
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio


def build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization,
                                    datasets_for_eval=[], data_path="",
                                    logger_prefix="OS2D.eval"):
    """构造用于评估的数据加载器。
    Args:
        cfg - 配置对象，评估在 cfg.eval.dataset_names 评估数据集上完成
        box_coder (Os2dBoxCoder)
        img_normalization (dict) - normalization to use, keys "mean" and "std" have lists of 3 floats each
        datasets_for_eval (list of DatasetOneShotDetection) - 用于评估的数据集，应包括训练集的子集和创建的数据集
        data_path (str) - 搜索数据集的根路径，如果提供将从配置创建评估数据集, '/xxx/os2d/data'
        logger_prefix (str) - prefix to ass to the logger outputs, 'OS2D.eval'
    Output:
        dataloaders_eval (list of DataloaderOneShotDetection) - the dataloaders
    """
    if data_path:
        # 检查是否正确提供了 eval_scales
        if len(cfg.eval.dataset_scales) == 1:
            eval_scales = cfg.eval.dataset_scales * len(cfg.eval.dataset_names)   #eg: [1280.0]
            eval_dataset_names = cfg.eval.dataset_names
        elif len(cfg.eval.dataset_names) == 1:
            eval_dataset_names = cfg.eval.dataset_names * len(cfg.eval.dataset_scales)
            eval_scales = cfg.eval.dataset_scales
        else:
            eval_scales = cfg.eval.dataset_scales
            eval_dataset_names = cfg.eval.dataset_names
        # 评估数据集的个数和评估数量需要一致，即1个评估数据集需要有1种评估数量
        assert len(eval_scales) == len(eval_dataset_names), "The number of values in eval_scales (have {0}: {1}) should be compatible with the number of values in eval_datasets_name (have {2}: {3})".format(len(eval_scales), eval_scales, len(eval_dataset_names), eval_dataset_names)
        # build all the eval datasets
        datasets_val = [build_dataset_by_name(data_path, dataset_name,
                                              eval_scale=eval_scale,
                                              cache_images=cfg.eval.cache_images,
                                              logger_prefix=logger_prefix)\
                        for dataset_name, eval_scale in zip(eval_dataset_names, eval_scales) if dataset_name]
    else:
        datasets_val = []

    # 如果提供，添加额外的训练数据集
    if len(datasets_for_eval) != 0:
        datasets_val = datasets_val + datasets_for_eval
        eval_scales = eval_scales + [d.eval_scale for d in datasets_for_eval]

    dataloaders_eval = []
    for dataset, eval_scale in zip(datasets_val, eval_scales):
        evaluation_scale = eval_scale / dataset.image_size
        pyramid_scales_eval = cfg.eval.scales_of_image_pyramid
        pyramid_scales_eval = [p * evaluation_scale for p in pyramid_scales_eval]
        
        dataloaders_eval.append(DataloaderOneShotDetection(dataset=dataset,
                                               box_coder=box_coder,
                                               batch_size=cfg.eval.batch_size,
                                               img_normalization=img_normalization,
                                               random_flip_batches=False,
                                               random_crop_size=None,
                                               random_color_distortion=False,
                                               gt_image_size=cfg.model.class_image_size,
                                               pyramid_scales_eval=pyramid_scales_eval,
                                               do_augmentation=False,
                                               logger_prefix=logger_prefix))
    return dataloaders_eval


def build_train_dataloader_from_config(cfg, box_coder, img_normalization,
                                       dataset_train=None, data_path="",
                                       logger_prefix="OS2D.train"):
    """构建用于训练的数据加载器。
    Args:
        cfg - config object, training is done on cfg.train.dataset_name dataset
        box_coder (Os2dBoxCoder)
        img_normalization (dict) - normalization to use, keys "mean" and "std" have lists of 3 floats each
        dataset_train (DatasetOneShotDetection) - 需要提供数据集对象或从配置创建此类对象的路径
        data_path (str) - 搜索数据集的根路径， eg: '/xxx/os2d/data'
        logger_prefix (str) - 日志：记录器输出的前缀
    Output:
        dataloader_train (DataloaderOneShotDetection) - the dataloader for training
        datasets_train_subset_for_eval (list of DatasetOneShotDetection) - 传递给评估数据加载器的训练集的子集
    """
    if dataset_train is None:
        assert data_path, "If explicit dataset_train is not provided one needs to provide a data_path to create one"
        dataset_train = build_dataset_by_name(data_path, cfg.train.dataset_name,
                                              eval_scale=cfg.train.dataset_scale,
                                              cache_images=cfg.train.cache_images,
                                              no_image_reading=not cfg.train.do_training)

    logger = logging.getLogger(logger_prefix+".dataloader")
    #随机裁剪大小
    random_crop_size = FeatureMapSize(w=cfg.train.augment.train_patch_width,
                                      h=cfg.train.augment.train_patch_height)
    evaluation_scale = dataset_train.eval_scale / dataset_train.image_size   #eg: 0.392
    #特征金字塔的缩放率
    pyramid_scales_eval = cfg.eval.scales_of_image_pyramid     #eg: [1.0]， eg: [0.5, 0.625, 0.8, 1, 1.2, 1.4, 1.6]
    pyramid_scales_eval = [p * evaluation_scale for p in pyramid_scales_eval]  #eg: [0.39215686274509803]

    dataloader_train = DataloaderOneShotDetection(dataset=dataset_train,
                                      box_coder=box_coder,
                                      batch_size=cfg.train.batch_size,
                                      class_batch_size=cfg.train.class_batch_size,
                                      img_normalization=img_normalization,
                                      random_flip_batches=cfg.train.augment.random_flip_batches,
                                      random_crop_size=random_crop_size,
                                      random_crop_scale=evaluation_scale,
                                      jitter_aspect_ratio=cfg.train.augment.jitter_aspect_ratio,
                                      scale_jitter=cfg.train.augment.scale_jitter,
                                      min_box_coverage=cfg.train.augment.min_box_coverage,
                                      random_color_distortion=cfg.train.augment.random_color_distortion,
                                      random_crop_class_images=cfg.train.augment.random_crop_class_images,
                                      gt_image_size=cfg.model.class_image_size,
                                      pyramid_scales_eval=pyramid_scales_eval,
                                      do_augmentation=True,
                                      mine_extra_class_images=cfg.train.augment.mine_extra_class_images,
                                      show_gt_boxes=cfg.visualization.train.show_gt_boxes_dataloader,
                                      logger_prefix=logger_prefix)

    if cfg.eval.train_subset_for_eval_size > 0:
        logger.info("Creating sub-training set of size {0} for evaluation".format(cfg.eval.train_subset_for_eval_size))
        datasets_train_subset_for_eval = [dataset_train.copy_subset(cfg.eval.train_subset_for_eval_size)]
    else:
        datasets_train_subset_for_eval = []
    return dataloader_train, datasets_train_subset_for_eval


class DataloaderOneShotDetection():
    """用于one-shot检测任务的数据加载器。
    此类包含数据集、数据增强和创建训练/评估批次的方法。
    For usage at training, see engine.train.
    For usage at evaluation, see engine.evaluate.
    """
    def __init__(self, dataset, box_coder, transform=None, transform_gt=None, batch_size=4, class_batch_size=None,
                 img_normalization=None, gt_image_size=224,
                 random_flip_batches=False, random_crop_size=None, random_crop_scale=1.0, random_color_distortion=False,
                 jitter_aspect_ratio=1.0, scale_jitter=1.0,
                 random_crop_class_images=False, min_box_coverage=0.7,
                 pyramid_scales_eval=(1, ),
                 do_augmentation=False,
                 mine_extra_class_images=False,
                 logger_prefix="OS2D",
                 show_gt_boxes=False):
        self.logger = logging.getLogger(f"{logger_prefix}.dataloader")
        self.dataset = dataset
        self.box_coder = box_coder
        self.img_normalization = img_normalization
        self.gt_image_size = gt_image_size
        self.hardnegs_per_imageid = None
        self.hardnegdata_per_imageid = None
        self.mine_extra_class_images = mine_extra_class_images
        self.show_gt_boxes = show_gt_boxes

        self.pyramid_scales_eval = pyramid_scales_eval
        self.num_pyramid_levels = len(self.pyramid_scales_eval)
        self.pyramid_box_inverse_transform = {}

        # 选择如何处理数据增强
        if do_augmentation:
            self.data_augmentation = DataAugmentation(random_flip_batches=random_flip_batches,
                                                      random_crop_size=random_crop_size,
                                                      random_crop_scale=random_crop_scale,
                                                      jitter_aspect_ratio=jitter_aspect_ratio,
                                                      scale_jitter=scale_jitter,
                                                      random_color_distortion=random_color_distortion,
                                                      random_crop_label_images=random_crop_class_images,
                                                      min_box_coverage=min_box_coverage)

            # 如果进行随机裁剪数据增强，则无需使用桶 - 所有图像将被裁剪为相同大小， eg: False
            self.use_buckets = False if random_crop_size is not None else True            
        else:
            self.data_augmentation = None
            self.use_buckets = True
        # eg: 4
        self.batch_size = batch_size
        self.max_batch_labels = class_batch_size  #eg: 15
        
        if self.dataset.have_images_read:
            # 通过创建具有相同大小的图像桶来设置批次
            self._create_buckets(merge_one_bucket=not self.use_buckets)

            # 从所有 groundtruth boxes 中挖掘所有 gt 图像
            if self.mine_extra_class_images:
                self._mine_extra_class_images()

    def get_name(self):
        return self.dataset.get_name()

    def get_eval_scale(self):
        return self.dataset.get_eval_scale()

    def _mine_extra_class_images(self):
        self.label_image_collection = {}
        # loop over buckets
        for ids_b in self.buckets:
            # loop over images
            for image_id in ids_b:
                img = self._get_dataset_image_by_id(image_id)
                boxes = self.get_image_annotation_for_imageid(image_id)
                assert boxes.has_field("labels")
                if not boxes.has_field("difficult"):
                    difficult_flag = boxes.add_field("difficult", torch.zeros(len(boxes), dtype=torch.bool))

                for box in boxes:
                    if not box.get_field("difficult").item():
                        # mine only non-difficult GT boxes
                        img_cropped, _, _, _ = crop(img, crop_position=box)
                        label = box.get_field("labels").item()
                        if label not in self.label_image_collection:
                            self.label_image_collection[label] = []
                        self.label_image_collection[label].append(img_cropped)

    def _create_buckets(self, merge_one_bucket=False):
        # 根据图像大小在桶中组织图像：在一批中我们只能看到来自同一个桶的图像
        if not merge_one_bucket:
            self.buckets = self.dataset.split_images_into_buckets_by_size()
        else:
            # 将所有图像放在一个桶中
            self.buckets = [list(self.dataset.image_size_per_image_id.keys())]  # copy all the image_ids - those are ints

        # 收集有关桶的信息
        self.num_buckets = len(self.buckets)
        self.bucket_sizes = [len(b) for b in self.buckets]
        self.num_batches_per_bucket = [math.ceil(bucket_size / self.batch_size) for bucket_size in self.bucket_sizes]
        self.num_batches = sum(self.num_batches_per_bucket)
        # setup the order of traversing buckets
        self.bucket_order = []
        for i_bucket in range(self.num_buckets):
            self.bucket_order.extend([ (i_bucket, i_batch) for i_batch in range(self.num_batches_per_bucket[i_bucket]) ])

    def shuffle(self, shuffle_buckets=True):
        self.bucket_order = [self.bucket_order[i] for i in torch.randperm(len(self.bucket_order))]
        if shuffle_buckets:
            for i_bucket, bucket in enumerate(self.buckets):
                num_items = len(bucket)
                self.buckets[i_bucket] = [bucket[i] for i in torch.randperm(num_items)]

    def _get_dataset_image_by_id(self, image_id):
        return self.dataset._get_dataset_image_by_id(image_id)

    def get_image_annotation_for_imageid(self, image_id):
        return self.dataset.get_image_annotation_for_imageid(image_id)

    def get_image_ids_for_batch_index(self, index):
        assert(index < self.num_batches)
        i_bucket, i_batch = self.bucket_order[index]
        image_ids = self.buckets[i_bucket][i_batch*self.batch_size:(i_batch+1)*self.batch_size]
        return image_ids

    def get_batch(self, index, use_all_labels=False):
        image_ids = self.get_image_ids_for_batch_index(index)
        return self._prepare_batch(image_ids, use_all_labels=use_all_labels)

    def _transform_image_to_pyramid(self, image_id, boxes=None,
                                          do_augmentation=True, hflip=False, vflip=False,
                                          pyramid_scales=(1,),
                                          mined_data=None ):
        img = self._get_dataset_image_by_id(image_id)  #读取图像的二进制像素
        img_size = FeatureMapSize(img=img)  #获取图像的大小, FeatureMapSize(w=3264, h=2448)
        # 是否使用数据增强， False
        do_augmentation = do_augmentation and self.data_augmentation is not None
        num_pyramid_levels = len(pyramid_scales)

        use_mined_crop = mined_data is not None
        if use_mined_crop:
            crop_position = mined_data["crop_position_xyxy"]

        if boxes is None:
            boxes = BoxList.create_empty(img_size)  #eg: BoxList(num_boxes=0, image_width=3264, image_height=2448, )
        mask_cutoff_boxes = torch.zeros(len(boxes), dtype=torch.bool)
        mask_difficult_boxes = torch.zeros(len(boxes), dtype=torch.bool)

        box_inverse_transform = TransformList()
        # 批级数据增强
        img, boxes = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip, 
                                                boxes=boxes,
                                                transform_list=box_inverse_transform)

        if use_mined_crop:
            # update crop_position_xyxy with the symmetries
            if hflip or vflip:
                _, crop_position = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip,
                                                                   boxes=crop_position)

        if do_augmentation:        
            if self.data_augmentation.do_random_crop:
                if not use_mined_crop:
                    img, boxes, mask_cutoff_boxes, mask_difficult_boxes = \
                        self.data_augmentation.random_crop(img,
                                                           boxes=boxes,
                                                           transform_list=box_inverse_transform)
                else:
                    img, boxes, mask_cutoff_boxes, mask_difficult_boxes = \
                        self.data_augmentation.crop_image(img, crop_position,
                                                          boxes=boxes,
                                                          transform_list=box_inverse_transform)

                img, boxes = transforms_boxes.resize(img, target_size=self.data_augmentation.random_crop_size,
                                                     random_interpolation=self.data_augmentation.random_interpolation,
                                                     boxes=boxes,
                                                     transform_list=box_inverse_transform)

            # color distortion
            img = self.data_augmentation.random_distort(img)

        random_interpolation = self.data_augmentation.random_interpolation if do_augmentation else False
        img_size = FeatureMapSize(img=img)
        pyramid_sizes = [ FeatureMapSize(w=int(img_size.w * s), h=int(img_size.h * s)) for s in pyramid_scales ]
        img_pyramid = []
        boxes_pyramid = []
        pyramid_box_inverse_transform = []
        for p_size in pyramid_sizes:
            box_inverse_transform_this_scale = copy.deepcopy(box_inverse_transform)
            p_img, p_boxes = transforms_boxes.resize(img, target_size=p_size, random_interpolation=random_interpolation,
                                                     boxes=boxes,
                                                     transform_list=box_inverse_transform_this_scale)
            
            pyramid_box_inverse_transform.append(box_inverse_transform_this_scale)
            img_pyramid.append( p_img )   #eg: <PIL.Image.Image image mode=RGB size=1280x960 at 0x7EFBBDBDBEB0>
            boxes_pyramid.append( p_boxes )   #eg: BoxList(num_boxes=0, image_width=1280, image_height=960, )

        transforms_th = [transforms.ToTensor()]
        if self.img_normalization is not None:
            transforms_th += [transforms.Normalize(self.img_normalization["mean"], self.img_normalization["std"])]

        for i_p in range(num_pyramid_levels):
            img_pyramid[i_p] = transforms.Compose(transforms_th)( img_pyramid[i_p] )
        # img_pyramid: 不同尺寸的特征金字塔向量，boxes_pyramid: 每个特征金字塔对应的感受野，
        return img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform

    def _transform_image(self, image_id, boxes=None, do_augmentation=True, hflip=False, vflip=False, mined_data=None):
        img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform = \
                self._transform_image_to_pyramid(image_id, boxes=boxes,
                                                 do_augmentation=do_augmentation, hflip=hflip, vflip=vflip,
                                                 pyramid_scales=(1,), mined_data=mined_data) 

        return img_pyramid[0], boxes_pyramid[0], mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform[0]

    def _transform_image_gt(self, img, do_augmentation=True, hflip=False, vflip=False, do_resize=True):
        do_augmentation = do_augmentation and self.data_augmentation is not None
        
        # 批级数据增强
        img, _ = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip, boxes=None, transform_list=None)

        if do_augmentation:
            # color distortion
            img = self.data_augmentation.random_distort(img)
            # random crop
            img = self.data_augmentation.random_crop_label_image(img)
            
        # resize image
        if do_resize:
            random_interpolation = self.data_augmentation.random_interpolation if do_augmentation else False

            # get the new size - while preserving aspect ratio
            size_old = FeatureMapSize(img=img)
            h, w = get_image_size_after_resize_preserving_aspect_ratio(h=size_old.h, w=size_old.w,
                target_size=self.gt_image_size)
            size_new = FeatureMapSize(w=w, h=h)

            img, _  = transforms_boxes.resize(img, target_size=size_new, random_interpolation=random_interpolation)
        
        transforms_th = [transforms.ToTensor()]
        if self.img_normalization is not None:
            transforms_th += [transforms.Normalize(self.img_normalization["mean"], self.img_normalization["std"])]
        img = transforms.Compose(transforms_th)(img)
        return img

    def unnorm_image(self, img):
        if self.img_normalization is not None:
            std_inv = [ 1.0 / s for s in self.img_normalization["std"]]
            mean_inv = [-m / s for m, s in zip(self.img_normalization["mean"], self.img_normalization["std"])]
            device = img.device
            img = transforms.Normalize(mean_inv, std_inv, inplace=False)(img.cpu())  # operation works only on CPU
            img = img.to(device=device)
        return img

    def get_class_images_and_sizes(self, class_ids, do_augmentation=False):
        if self.mine_extra_class_images and do_augmentation:
            # select random label image if several are mined
            class_images = []
            for class_id in class_ids:
                if class_id in self.label_image_collection:
                    num_mined = len(self.label_image_collection[class_id])
                    random_int = torch.randint(num_mined+1, (1,), dtype=torch.long)
                    if random_int == 0:
                        # use the original image
                        class_image = self.dataset.gt_images_per_classid[class_id]    
                    else:
                        # use the selected mined image
                        class_image = self.label_image_collection[class_id][random_int - 1]
                else:
                    # nothing was mined for this class
                    class_image = self.dataset.gt_images_per_classid[class_id]
                class_images.append(class_image)
        else:
            class_images = [self.dataset.gt_images_per_classid[class_id] for class_id in class_ids]
        class_image_sizes = [FeatureMapSize(img=img) for img in class_images]
        return class_images, class_image_sizes
        
    def get_all_class_images(self, do_resize=True):
        class_ids = self.dataset.get_class_ids()
        class_ids = sorted(list(class_ids))
        class_images, class_image_sizes = self.get_class_images_and_sizes(class_ids, do_augmentation=False)  #获取PIL图像和图像的特征图大小
        batch_class_images = [self._transform_image_gt(img, do_augmentation=False, do_resize=do_resize) for img in class_images]

        # 要具有适当的维度，只需将维度零添加到所有图像
        batch_class_images = [img.unsqueeze(0) for img in batch_class_images]
        return batch_class_images, class_image_sizes, class_ids

    def get_class_ids_for_image_ids(self, image_ids):
        return self.dataset.get_class_ids_for_image_ids(image_ids)

    def make_iterator_for_all_images(self, batch_size, num_random_pyramid_scales=0):
        # 再次创建桶不要打乱或重新打乱用于训练的桶
        buckets_ids = self.dataset.split_images_into_buckets_by_size()
        
        batch_size = max(len(ids) for ids in buckets_ids) if batch_size is None else batch_size
        num_batches = (sum( int(math.ceil(len(ids) / batch_size)) for ids in buckets_ids))
        i_batch = 0

        for ids_b in buckets_ids:
            size_b = len(ids_b)
            
            # batch images
            for batch_start in range(0, size_b, batch_size):
                self.logger.info("图像批次 {0} 中包含的图像个数是 {1}".format(i_batch, num_batches))
                i_batch += 1
                batch_ids = ids_b[batch_start : batch_start + batch_size]

                img_pyramid_all_images = []
                pyramid_box_inverse_transform_all_images = []
                initial_img_size_this_batch = []

                # 为此批次选择金字塔比例尺
                if not num_random_pyramid_scales:
                    pyramid_scales = self.pyramid_scales_eval
                else:
                    min_scale = min(self.pyramid_scales_eval)
                    max_scale = max(self.pyramid_scales_eval)
                    pyramid_scales = [torch.rand(1).item() * (max_scale - min_scale) + min_scale  for i in range(num_random_pyramid_scales)]

                for image_id in batch_ids:
                    img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform = \
                                self._transform_image_to_pyramid(image_id,
                                                                boxes=None,
                                                                do_augmentation=False, pyramid_scales=pyramid_scales)
                    img_pyramid_all_images.append(img_pyramid)
                    pyramid_box_inverse_transform_all_images.append(pyramid_box_inverse_transform)
                    initial_img_size_this_batch.append( self.dataset.get_image_size_for_image_id(image_id) )

                # batch pyramids
                pyramids_this_batch = []
                transforms_this_batch = pyramid_box_inverse_transform_all_images
                for i_p in range(len(pyramid_scales)):
                    pyramids_this_batch.append( torch.stack( [ p_one_image[i_p] for p_one_image in img_pyramid_all_images ], 0) )

                yield batch_ids, pyramids_this_batch, transforms_this_batch, initial_img_size_this_batch

    @staticmethod
    def convert_label_ids_global_to_local(label_ids_global, class_ids):
        label_ids_local = [] # local indices w.r.t. batch_class_images
        if label_ids_global is not None:
            for label_id in label_ids_global:
                label_id = label_id.item()
                label_ids_local.append( class_ids.index(label_id) if label_id in class_ids else -1 )
        label_ids_local = torch.tensor(label_ids_local, dtype=torch.long)
        return label_ids_local

    @staticmethod
    def update_box_labels_to_local(boxes, class_ids):
        label_ids_global = boxes.get_field("labels")  #tensor([30, 30, 30])有3个bbox，每个的类别id
        # label_ids_local: tensor([0, 0, 0])  换成本地标签
        label_ids_local = DataloaderOneShotDetection.convert_label_ids_global_to_local(label_ids_global, class_ids)
        boxes.add_field("labels", label_ids_local)
    
    def set_hard_negative_data(self, hardnegdata_per_imageid):
        self.hardnegdata_per_imageid = copy.deepcopy(hardnegdata_per_imageid)
    
    def _prepare_batch(self, image_ids, use_all_labels=False):
        batch_images = []
        batch_class_images = []
        batch_loc_targets = []
        batch_class_targets = []
        
        # flag to use hard neg mining
        use_mined_data = self.hardnegdata_per_imageid is not None
        # select which mined boxes to use
        if use_mined_data:
            # for half of the images select hard positives, for half - hard negatives
            # the order of images in a batch is random, so no need to randomize here
            batch_size = len(image_ids)
            num_neg_patches = batch_size // 2
            role_to_select = ["neg"] * num_neg_patches + ["pos"] * (batch_size - num_neg_patches)
            mined_data = {}
            for image_id, role in zip(image_ids, role_to_select):
                mined_data_for_image = self.hardnegdata_per_imageid[image_id]
                # filter for the correct role
                mined_data_for_image = [d for d in mined_data_for_image if d["role"][:len(role)] == role]
                if len(mined_data_for_image) == 0:
                    mined_data_for_image = self.hardnegdata_per_imageid[image_id]
                assert len(mined_data_for_image) > 0, "Could not find mined {0} for image {1}".format(role, image_id)
                # select random element
                i_rand = torch.randint(len(mined_data_for_image), (1,), dtype=torch.long).item()
                mined_data[image_id] = mined_data_for_image[i_rand]
                # self.logger.info("Image {0}, mined data: {1}".format(image_id,  mined_data[image_id]))

        # collect labels for this batch
        batch_data =  self.dataset.get_dataframe_for_image_ids(image_ids)

        if not use_all_labels:
            class_ids = batch_data["classid"].unique()
            # select labels for mined hardnegs
            if use_mined_data:
                # select labels that are compatible with mining
                mined_labels = [mined_data[image_id]["label_global"] for image_id in mined_data]
            else:
                mined_labels = []

            # randomly prune label images if too many
            max_batch_labels = self.max_batch_labels if self.max_batch_labels is not None else class_ids.size + len(mined_labels) + 1
            
            class_ids = np.unique(class_ids)
            np.random.shuffle(class_ids)
            class_ids = class_ids[:max_batch_labels - len(mined_labels)]
            
            class_ids = np.unique(np.concatenate((class_ids, np.array(mined_labels).astype(class_ids.dtype)), axis=0))
        else:
            class_ids = self.dataset.get_class_ids()
        class_ids = sorted(list(class_ids))
        
        # decide on batch level data augmentation
        if self.data_augmentation is not None:
            batch_vflip = random.random() < 0.5 if self.data_augmentation.batch_random_vflip else False
            batch_hflip = random.random() < 0.5 if self.data_augmentation.batch_random_hflip else False
        else:
            batch_vflip = False
            batch_hflip = False

        # prepare class images
        num_classes = len(class_ids)
        class_images, class_image_sizes = self.get_class_images_and_sizes(class_ids, do_augmentation=True)
        batch_class_images = [self._transform_image_gt(img, hflip=batch_hflip, vflip=batch_vflip) for img in class_images]
        # get the image sizes after resize in self._transform_image_gt, format - width, height
        class_image_sizes = [FeatureMapSize(img=img) for img in batch_class_images]

        # prepare images and boxes
        img_size = None
        batch_box_inverse_transform = []
        batch_boxes = []
        batch_img_size = []
        for image_id in image_ids:
            # get annotation
            boxes = self.get_image_annotation_for_imageid(image_id)

            # convert global indices to local
            # if use_global_labels==False then local indices will be w.r.t. labels in this batch
            # if use_global_labels==True then local indices will be w.r.t. labels in the whole dataset (not class_ids)
            self.update_box_labels_to_local(boxes, class_ids)
            
            # prepare image and boxes: convert image to tensor, data augmentation: some boxes might be cut off the image
            image_mined_data = None if not use_mined_data else mined_data[image_id]
            img, boxes, mask_cutoff_boxes, mask_difficult_boxes, box_inverse_transform = \
                     self._transform_image(image_id, boxes, hflip=batch_hflip, vflip=batch_vflip, mined_data=image_mined_data)

            # mask_difficult_boxes is set True for boxes that are largely chopped off, those are not used for training
            if boxes.has_field("difficult"):
                old_difficult = boxes.get_field("difficult")
                boxes.add_field("difficult", old_difficult | mask_difficult_boxes)
            boxes.get_field("labels")[mask_cutoff_boxes] = -2

            # vizualize groundtruth images and boxes - to debug data augmentation
            if self.show_gt_boxes and self.data_augmentation is not None:
                visualizer.show_gt_boxes(image_id, boxes, class_ids, self, image_to_show=img)

            # check image size in this batch
            if img_size is None:
                img_size = FeatureMapSize(img=img)
            else:
                assert img_size == FeatureMapSize(img=img), "Images in a batch should be of the same size"

            loc_targets, class_targets = self.box_coder.encode(boxes, img_size, num_classes)
            batch_loc_targets.append(loc_targets)
            batch_class_targets.append(class_targets)
            batch_images.append(img)
            batch_box_inverse_transform.append( [box_inverse_transform] )
            batch_boxes.append(boxes)
            batch_img_size.append(img_size)
            
        # stack data
        batch_images = torch.stack(batch_images, 0)
        batch_loc_targets = torch.stack(batch_loc_targets, 0)
        batch_class_targets = torch.stack(batch_class_targets, 0)

        return batch_images, batch_class_images, batch_loc_targets, batch_class_targets, class_ids, class_image_sizes, \
               batch_box_inverse_transform, batch_boxes, batch_img_size
    
    def __len__(self):
        return self.num_batches
