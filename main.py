import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask
from flask import request
from PIL import Image
from waitress import serve

import os

app = Flask(__name__)

image = keras.preprocessing.image
preprocess_input = keras.applications.inception_v3.preprocess_input
vgg16 = keras.applications.VGG16(include_top=False)
reconstructed_model = keras.models.load_model("model_v1")

def prepare_image(src_img):
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

@app.route('/infer', methods = ['POST'])
def infer():
  f = request.files['file']
  img = Image.open(f)
  prepared = prepare_image(img)
  infered = reconstructed_model(prepared)

  # Dimension 0 is batch image, dimension 1 is the result float
  # infered is in Tensor format, convert to normal Python float
  infer_float = float(infered[0][0])

  is_not_safe = False if infer_float > 0.5 else True

  res = {
    "safety_score": infer_float,
    "depict_nudity": is_not_safe
  }
  
  return res

if __name__ == '__main__':
  use_port = os.getenv("PORT") if os.getenv("PORT") is not None else 5003
  # app.run(debug = True, port=use_port)
  serve(app, host='0.0.0.0', port=use_port)
