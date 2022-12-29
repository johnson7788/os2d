#!/usr/bin/env python
# coding: utf-8

# 这是一个演示，说明了 OS2D 方法在一张图像上的应用。
# 该演示假设 OS2D API 运行在您机器的 80 端口，请按照以下说明进行操作：[Run OS2D as Service](./FASTAPI.md)。

from PIL import Image, ImageDraw
import os
import base64
import json
import requests


# ## Running Docker container

os.system('docker run -d --rm     --name os2d     -p 80:80     -v $(pwd):/workspace     os2d:latest     uvicorn app:app --port 80 --host 0.0.0.0')


# ## Load images


input_image_path = 'data/demo/input_image.jpg'
first_query_image_path = 'data/demo/class_image_0.jpg'
second_query_image_path = 'data/demo/class_image_1.jpg'


with open(input_image_path, 'rb') as i, open(first_query_image_path, 'rb') as fq, open(second_query_image_path, 'rb') as sq:
    input_image = base64.b64encode(i.read()).decode('utf-8')
    first_query_image = base64.b64encode(fq.read()).decode('utf-8')
    second_query_image = base64.b64encode(sq.read()).decode('utf-8')


# ## Build request body


body = json.dumps({
    'image': {'content': input_image},
    'query': [
        {'content': first_query_image},
        {'content': second_query_image}
        ]
})


# ## Send request POST


# http://0.0.0.0:80 -> Localhost port 80
res = requests.post("http://0.0.0.0:80/detect-all-instances", data=body)



input_image = Image.open(input_image_path)
im_w, im_h = input_image.size



boxes = [[(box[0] * im_w, box[1] * im_h), (box[2] * im_w, box[3] * im_h) ] for box in res.json()['bboxes']]



im = ImageDraw.Draw(input_image)
for box in boxes:
    im.rectangle(box, outline='yellow', width=3)
input_image

