from flask import Flask, render_template, Response, request
import requests
from importlib import import_module

import camera_opencv
import webbrowser
import cv2
import os

camera = cv2.VideoCapture(0)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_model_inference = "Model 1"
    # selected_model_inference = request.form["flexModelSelection"]
    return render_template('index.html', selected_model_inference=selected_model_inference)


# @app.route("/model_selection", methods=["POST"])
# def model_selection():
#     model = request.form["flexModelSelection"]
#     return f"Model is {model}"


def gen(camera):
    """Video streaming generator function."""
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg',frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', threaded=True)
