#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/1/29 15:45
# @File  : data_utils.py
# @Author:
# @Desc  : 一些数据操作的函数
import os
import pandas as pd
from tqdm import tqdm
import random
import shutil
from utils import read_data

def generate_cosmetic_data():
    """
    生成化妆品的数据集
    """
    data_path = "../data/cosmetic"
    # 创建标注的类别目录
    classes_path = os.path.join(data_path, "classes")
    # 标注文件的名字
    annotation_file = os.path.join(classes_path, "cosmetic.csv")
    # 创建类别的图片目录
    classes_images_path = os.path.join(classes_path, "images")
    # 原始图片目录
    src_images_path = os.path.join(data_path, "src")
    print(f"删除旧的数据集")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    print(f"创建数据集目录")
    if not os.path.exists(classes_images_path):
        os.makedirs(classes_images_path)
    if not os.path.exists(src_images_path):
        os.makedirs(src_images_path)
    # 读取数据
    source_data = read_data()
    # 保存数据到指定目录
    # 保存到csv文件的列, gtbboxid,classid,imageid,lx,rx,ty,by,difficult,split, imagefilename, classfilename
    # 所有的商品
    data = []
    class2id = {}
    for idx, one in enumerate(tqdm(source_data)):
        if idx % 200 == 0:
            print(f"完成了: {idx} 条")
        gtbboxid = idx
        product = one["product"]
        if product in class2id:
            classid = class2id[product]
        else:
            classid = len(class2id)
            class2id[product] = classid
        imageid = one['md5']
        # bbox是左上角的点和右下角的点, 需要换成百分比格式
        lx, ty, rx, by = one['bbox']
        difficult = random.choice([0,1])
        split = random.choices(['train', 'val'], [0.8, 0.2], k=1)[0]
        imagefilename_path = one["path"]
        imagefilename = os.path.basename(imagefilename_path)
        # 拷贝图片到固定目录
        src_filepath = os.path.join(src_images_path, imagefilename)
        shutil.copy(src=imagefilename_path,dst=src_filepath)
        classfilename_path = one["product_path"]
        classfilename = os.path.basename(classfilename_path)
        # 拷贝类别图片到固定目录
        class_filepath = os.path.join(classes_images_path, classfilename)
        shutil.copy(src=classfilename_path,dst=class_filepath)
        one_data = {
            "gtbboxid":gtbboxid,
            "classid":classid,
            "imageid": imageid,
            "lx": lx,
            "rx": rx,
            "ty": ty,
            "by": by,
            "difficult": difficult,
            "split":split ,
            "imagefilename": imagefilename,
            "classfilename": classfilename
        }
        data.append(one_data)
    data_df = pd.DataFrame(data)
    data_df.to_csv(annotation_file)
    print(f"生成comestic数据完成")

if __name__ == '__main__':
    generate_cosmetic_data()
