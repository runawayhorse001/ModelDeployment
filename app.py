# main reference:
# https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production/blob/master/README.md
# you could also generate a skeleton from scratch via
# http://flask-appbuilder.readthedocs.io/en/latest/installation.html

# Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
# HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine
# for you automatically.
# requests are objects that flask handles (get set post, etc)

from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
# from scipy.misc import imsave, imread, imresize
from PIL import Image
#
import io
import re
import cv2
# for matrix math
import numpy as np
# for importing our keras model
import keras.models

# for convert base64 string to image
import base64

# system level operations (like loading files)
import sys
# for reading operating system data
import os

from model import load, convertImage, transform, transform2

# initalize our flask app
app = Flask(__name__)

# initialize these variables
model, graph = load()

@app.route('/')
def index():
    """
    initModel()
    render out pre-built HTML file right on the index page
    :return:
    """
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    """
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    #get the raw data format of the image
    :return:
    """
    # request data
    imgData = request.get_data()
    #
    # transform data to feature matrix
    # method 1: str --> image --> data
    # x = transform(imgData)

    # method 2: str--> data
    x = transform2(imgData)
    # print("debug2")
    # in our computation graph
    with graph.as_default():
        # perform the prediction
        out = model.predict(x)
        # print(out)
        # print(np.argmax(out, axis=1))
        # print("debug3")
        # convert the response to a string
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == "__main__":
    # decide what port to run the app in
    # port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    # app.run(host='0.0.0.0', port=port)
    # optional if we want to run in debugging mode
    # app.run(debug=False)
    app.run()
