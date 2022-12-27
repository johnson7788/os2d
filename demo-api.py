#!/usr/bin/env python
# coding: utf-8

# This is a demo illustrating an application of the OS2D method on one image.
# This demo assumes the OS2D API is running at port 80 of your machine, please follow instructions from: [Run OS2D as Service](./FASTAPI.md).

# In[ ]:


from PIL import Image, ImageDraw
import base64
import json
import requests


# ## Running Docker container

# In[ ]:


get_ipython().system('docker run -d --rm     --name os2d     -p 80:80     -v $(pwd):/workspace     os2d:latest     uvicorn app:app --port 80 --host 0.0.0.0')


# ## Load images

# In[ ]:


input_image_path = 'data/demo/input_image.jpg'
first_query_image_path = 'data/demo/class_image_0.jpg'
second_query_image_path = 'data/demo/class_image_1.jpg'


# In[ ]:


with open(input_image_path, 'rb') as i, open(first_query_image_path, 'rb') as fq, open(second_query_image_path, 'rb') as sq:
    input_image = base64.b64encode(i.read()).decode('utf-8')
    first_query_image = base64.b64encode(fq.read()).decode('utf-8')
    second_query_image = base64.b64encode(sq.read()).decode('utf-8')


# ## Build request body

# In[ ]:


body = json.dumps({
    'image': {'content': input_image},
    'query': [
        {'content': first_query_image},
        {'content': second_query_image}
        ]
})


# ## Send request POST

# In[ ]:


# http://0.0.0.0:80 -> Localhost port 80
res = requests.post("http://0.0.0.0:80/detect-all-instances", data=body)


# In[ ]:


input_image = Image.open(input_image_path)
im_w, im_h = input_image.size


# In[ ]:


boxes = [[(box[0] * im_w, box[1] * im_h), (box[2] * im_w, box[3] * im_h) ] for box in res.json()['bboxes']]


# In[ ]:


im = ImageDraw.Draw(input_image)
for box in boxes:
    im.rectangle(box, outline='yellow', width=3)
input_image

