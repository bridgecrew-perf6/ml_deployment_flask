from re import I
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask
from flask import request
from PIL import Image
from waitress import serve
import pandas as pd
import requests
from io import BytesIO


import os

app = Flask(__name__)

image = keras.preprocessing.image
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

nudity_model = keras.models.load_model("nudity_v2_mobilenet")

landmark_model = keras.models.load_model("landmark_v1_model/indonesia_landmark_model_MobileNetV2_100epoch.h5")
landmark_classes = pd.read_csv('landmark_v1_model/label_name.csv', header=0)

def prepare_image_mobilenet(src_img):
  """
  Convert PIL image (In memory) to RGB image (Sometimes the input image is in RGBA)
  then resize to 224x224 (Input size of the model)
  and preprocess further using keras
  """

  img = src_img.resize((224,224))
  img = img.convert('RGB')
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  return preprocess_input(img)

@app.route('/nudity', methods = ['POST'])
def inferNudity():
  if ('file' in request.files):
    f = request.files['file']
    img = Image.open(f)
  elif ('url' in request.json):
    response = requests.get(request.json['url'])
    img = Image.open(BytesIO(response.content))

  prepared = prepare_image_mobilenet(img)
  infered = nudity_model(prepared)

  # Dimension 0 is batch image, dimension 1 is the result float
  # infered is in Tensor format, convert to normal Python float
  infer_float = float(infered[0][0])

  is_not_safe = False if infer_float > 0.5 else True

  res = {
    "safety_score": infer_float,
    "depict_nudity": is_not_safe
  }
  
  return res

def prepare_image_landmark(src_img):
  """
  Convert PIL image (In memory) to RGB image (Sometimes the input image is in RGBA)
  then resize to 300xe300 (Input size of the model)
  and preprocess further using keras
  """

  img = src_img.resize((300,300))
  img = img.convert('RGB')
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  return preprocess_input(img)

@app.route('/landmark', methods = ['POST'])
def inferLandmark():
  if ('file' in request.files):
    f = request.files['file']
    img = Image.open(f)
  elif ('url' in request.json):
    response = requests.get(request.json['url'])
    img = Image.open(BytesIO(response.content))

  prepared = prepare_image_landmark(img)
  infered = landmark_model(prepared)

  sorted_result = np.argsort(-infered[0])

  result = []
  for i, it in enumerate(sorted_result):
    result.append({
      "id": int(it),
      "rank": int(i),
      "name": landmark_classes.iloc[int(it)][1]
    })

  res = {
    "scores": result,
  }  

  return res

if __name__ == '__main__':
  use_port = int(os.environ.get("PORT", 8080))
  print(f"App run on port {use_port}")
  app.run(debug = True, host='0.0.0.0', port=use_port)
  # serve(app, host='0.0.0.0', port=use_port)
