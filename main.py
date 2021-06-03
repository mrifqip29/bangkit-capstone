from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from flask import Flask, request, flash, redirect
from PIL import Image
import flask
import numpy as np
import pandas as pd
import tensorflow as tf
import io

# from keras.models import load_model

# instantiate flask
app = Flask(__name__)

#  load model, and pass in the custom metric
global graph
graph = tf.compat.v1.get_default_graph()
model = load_model('traditional_cake.h5')


def prepare_image(image, target_size):
    # resize the input image and preprocess it
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.vstack([image])

    # return the processed image
    return image


class_names = ['kue_dadar_gulung', 'kue_kastengel', 'kue_klepon',
               'kue_lapis', 'kue_lumpur', 'kue_putri_salju', 'kue_risoles', 'kue_serabi']


@app.route("/")
def hello():
    return "Hello, Bangkit!"


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # indicate that the request was a success
            data["success"] = True

            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            data["filename"] = flask.request.files["image"].filename

            # preprocess the image and prepare it for classification
            resizedImage = prepare_image(image, target_size=(150, 150))

            classes = model.predict(resizedImage, batch_size=10)

            x = np.where(classes[0] == 1)[0][0]

            data["prediction"] = class_names[x]

    print("data: {}".format(data))
    print("data_type: {}".format(type(data)))

    return flask.jsonify(data)


@app.route("/json", methods=["POST"])
def json():
    data = {"success": False}

    if flask.request.method == "POST":
        data["input"] = request.get_json()

        # indicate that the request was a success
        data["success"] = True

    print("data: {}".format(data))
    print("data_type: {}".format(type(data)))

    return flask.jsonify(data)


@app.route("/text", methods=["POST"])
def text():
    data = {"success": False}

    if flask.request.method == "POST":
        data["input"] = request.form.get('image')

        # indicate that the request was a success
        data["success"] = True

    print("data: {}".format(data))
    print("data_type: {}".format(type(data)))

    return flask.jsonify(data)


@app.route("/coba", methods=["POST"])
def coba():
    data = {"success": False}

    if flask.request.method == "POST":
        # check if the post has "image" file part
        if 'image' not in request.files:
            data['message'] = "No image part"
            return flask.jsonify(data), 404

        file = request.files['image']

        if file.filename == '':
            data['message'] = "No selected image"
            return flask.jsonify(data), 404

        if file:
            # read the image in PIL format
            image = file.read()
            image = Image.open(io.BytesIO(image))

            data["filename"] = file.filename

            # preprocess the image and prepare it for classification
            resizedImage = prepare_image(image, target_size=(150, 150))

            classes = model.predict(resizedImage, batch_size=10)

            x = np.where(classes[0] == 1)[0][0]

            data["prediction"] = class_names[x]
            data["x_value"] = str(classes)

            # indicate that the request was a success
            data["success"] = True

    print("data: {}".format(data))
    print("data_type: {}".format(type(data)))

    return flask.jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
