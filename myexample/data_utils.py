#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2023/1/29 15:45
# @File  : data_utils.py
# @Author:
# @Desc  : 一些数据操作的函数
import os
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
    data = read_data()


if __name__ == '__main__':
    generate_cosmetic_data()
