import numpy as np
import keras.models
# for regular expressions, saves time dealing with string data
import re
import io
from PIL import Image
#
import io
# for convert base64 string to image
# import cv2 library for saving, reading, and resizing images
import cv2
import base64
from keras.models import model_from_json
from scipy.misc import imread, imresize, imshow
import tensorflow as tf


def load():
    try:
        # load woeights into new model
        json_file = open('./model/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./model/model.h5")
        print("Loaded Model from disk")
    except:
        raise ValueError("Can not find model, please run train.py to train model!!!")

    # compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # loss,accuracy = model.evaluate(X_test,y_test)
    # print('loss:', loss)
    # print('accuracy:', accuracy)
    graph = tf.get_default_graph()

    return loaded_model, graph


# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgData1 = imgData1.decode("utf-8")
    imgstr = re.search(r'base64,(.*)', imgData1).group(1)
    # print(imgstr)
    imgstr_64 = base64.b64decode(imgstr)
    with open('output/output.png', 'wb') as output:
            output.write(imgstr_64)


def transform(imgData):
    #
    convertImage(imgData)
    # read the image into memory
    x = cv2.imread('output/output.png', 0)
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    # make it the right size
    x = cv2.resize(x, (28, 28))
    # imshow(x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)

    return x


def transform2(imgData):
    image_b64 = imgData.decode("utf-8").split(",")[1]
    binary = base64.b64decode(image_b64)
    image = np.asarray(bytearray(binary), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    x = np.invert(img)
    # make it the right size
    x = cv2.resize(x, (28, 28))
    # imshow(x)
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1, 28, 28, 1)

    return x

