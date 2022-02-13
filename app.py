from flask import Flask
from flask.helpers import url_for
from flask_login import LoginManager
from flask import session, request
from werkzeug.utils import redirect
import os
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg16 import preprocess_input


def eye(path):
    img_rows = 224
    img_cols = 224
    num_channel = 3
    img_data_list = []
    model1 = load_model("./Eye_model.h5")
    category = ["Bulging Eyes", "Cataracts", "Crossed Eyes", "Glaucoma", "Uveitis"]
    input_img = cv2.imread(path)
    input_img_resize = cv2.resize(input_img, (img_rows, img_cols))
    img_data_list.append(input_img_resize)
    img_data = np.array(img_data_list)  # convert images in numpy array
    img_data = img_data.astype("float32")
    img_data /= 255
    return category[np.argmax(model1.predict(img_data)[0])]


def lung_cancer(path):
    my_image = load_img(path, target_size=(224, 224))

    # preprocess the image
    my_image = img_to_array(my_image)
    my_image = my_image.reshape(
        (1, my_image.shape[0], my_image.shape[1], my_image.shape[2])
    )
    my_image = preprocess_input(my_image)

    # make the prediction
    model = load_model("./ChestCancer_model.h5")
    prediction = model.predict(my_image)
    categories = [
        "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
        "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
        "normal",
        "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
    ]
    return categories[np.argmax(prediction)]


def brain_tumor(path):
    categories = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    model1 = load_model("mri_model.h5")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (128, 128))
    X = []
    X.append(img)
    X = np.array(X).reshape(-1, 128, 128)
    X = X / 255.0
    X = X.reshape(-1, 128, 128, 1)
    y_pred = model1.predict(X)
    return categories[np.argmax(y_pred)]


app = Flask(__name__)
app.secret_key = b"P1ec81bf6o`416b8eeb4980ba9b5"

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@app.route("/")
def index():
    if "username" in session:
        return "logged in as " + session["username"]
    return "Not logged in"


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        session["username"] = request.form["username"]
        redirect(url_for("index"))
    return """
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    """


UPLOAD_FOLDER = "./upload"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/eye", methods=["GET", "POST"])
def upload_file_eye():
    if request.method == "POST":
        if "file1" not in request.files:
            return "there is no file1 in form!"
        file1 = request.files["file1"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file1.save(path)
        return eye(path)

    return """
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    """


@app.route("/lung_cancer", methods=["GET", "POST"])
def upload_file_lung():
    if request.method == "POST":
        if "file1" not in request.files:
            return "there is no file1 in form!"
        file1 = request.files["file1"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file1.save(path)
        return lung_cancer(path)

    return """
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    """


@app.route("/brain_tumor", methods=["GET", "POST"])
def upload_file_brain():
    if request.method == "POST":
        if "file1" not in request.files:
            return "there is no file1 in form!"
        file1 = request.files["file1"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file1.filename)
        file1.save(path)
        return brain_tumor(path)

    return """
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    """


@app.route("/logout")
def logout():
    # remove the username from the session if it's there
    session.pop("username", None)
    return redirect(url_for("index"))


app.run(debug=True)
