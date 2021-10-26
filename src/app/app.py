from flask import Flask, render_template, Response, request
import requests
from importlib import import_module
import io
import base64
import queue

import camera_opencv
import webbrowser
import cv2
import os
import numpy as np
from collections import OrderedDict
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from time import time
import torch
from torch.autograd import Variable
from torchvision.transforms import *
from DemoModel import FullModel
from torch import nn
import transforms as t
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import json
import time
import jsonify

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


with open("jester-v1-labels.txt", "r") as fh:
    gesture_labels = fh.read().splitlines()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 48)

confidence_queue = queue.Queue(maxsize=10)

app = Flask(__name__)

@app.route('/get_model_selected', methods=['POST'])
def get_model_selected():
    try:
        model_selected = request.form.get('model_selected')
        print(model_selected)
        return jsonify(model_selected=model_selected)
    except Exception as e:
        return str(e)


@app.route('/', defaults={'selected_model_name': None}, methods=['GET', 'POST'])
@app.route("/<any(Demo_Model_1_20BNJester, Google_MediaPipe_Holistic_Model, Model_3):selected_model_name>")
def index(selected_model_name):
    gesture_recognition_state  = request.args.get('gesture_recognition_state', None)
    if gesture_recognition_state == None:
        gesture_recognition_state = "off"
    if gesture_recognition_state == "on" and selected_model_name == None:
        selected_model_name = "Demo_Model_1_20BNJester"
    return render_template("index.html",
                            selected_model_name=selected_model_name,
                            gesture_recognition_state=gesture_recognition_state)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return 'You want path: %s' % path


def gen(camera):
    """Video streaming generator function."""

    while True:
        success, frame = camera.read()
        cv2.flip(frame, 1, frame)
        
        if not success:
            break
        else:
            
            ret, buffer = cv2.imencode('.jpg', frame)#bg)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


import collections
import time

class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return round(len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0]), 2)
        else:
            return 0.0

def Demo_Model_1_20BNJester_gen(camera):
    """Video streaming generator function for Demo_Model_1_20BNJester."""
    # fig, ax = plt.subplots()
    # Set up some storage variables
    seq_len = 16
    value = 0
    imgs = []
    pred = 8
    top_3 = [9,8,7]
    out = np.zeros(10)

    model = FullModel(batch_size=1, seq_lenght=16)
    loaded_dict = torch.load('demo.ckp')
    model.load_state_dict(loaded_dict)
    model = model.cuda()
    model.eval()

    std, mean = [0.2674,  0.2676,  0.2648], [ 0.4377,  0.4047,  0.3925]
    transform = Compose([
        t.CenterCrop((96, 96)),
        t.ToTensor(),
        t.Normalize(std=std, mean=mean),
    ])

    s = time.time()
    n = 0
    hist = []
    mean_hist = []
    setup = True
    
    cooldown = 0
    eval_samples = 2
    num_classes = 27

    score_energy = torch.zeros((eval_samples, num_classes))

    fps_a = FPS()
    fps_d = FPS()

    while True:
        success, frame = camera.read()
        cv2.flip(frame, 1, frame)
        # print(f"fps_all: {fps_a()}")
        
        if not success:
            break
        else:
            # image = cv2.rectangle(frame, (5,5), (300,300), (0,255,0), 2)

            resized_frame = cv2.resize(frame, (160, 120))
            pre_img = Image.fromarray(resized_frame.astype('uint8'), 'RGB')

            img = transform(pre_img)

            if n % 4 == 0:
                imgs.append(torch.unsqueeze(img, 0))

            # Get model output prediction
            if len(imgs) == 16:

                # print(f"detection_iter_per_sec: {fps_d()}")

                data = torch.cat(imgs).cuda()
                output = model(data.unsqueeze(0))
                out = (torch.nn.Softmax(dim=1)(output).data).cpu().numpy()[0]
                if len(hist) > 300:
                    mean_hist  = mean_hist[1:]
                    hist  = hist[1:]
                
                # this is straight cheating.
                out[-2:] = [0,0]
                # Softmax should sum to 1.
                print(sum(out))

                hist.append(out)

                score_energy = torch.tensor(hist[-eval_samples:])
                curr_mean = torch.mean(score_energy, dim=0)
                mean_hist.append(curr_mean.cpu().numpy())
                #value, indice = torch.topk(torch.from_numpy(out), k=1)
                value, indice = torch.topk(curr_mean, k=1)
                indices = np.argmax(out)
                top_3 = out.argsort()[-3:]
                if cooldown > 0:
                    cooldown = cooldown - 1
                if value.item() > 0.6 and indices < 25 and cooldown == 0: 
                    print('Gesture:', gesture_labels[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
                    cooldown = 16 
                pred = indices
                imgs = imgs[1:]

                # send predictions to plotting thread
                try:
                    confidence_queue.put_nowait(out)
                except queue.Full as e:
                    print("WARNING: gesture scores filled output queue Filled")
                    pass

            n += 1
            bg = np.full((480, 640, 3), 15, np.uint8)
            bg[:480, :640] = frame

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # if value > 0.6:
            #     cv2.putText(bg, ges[pred],(20,465), font, 1,(0,255,0),2)
            # cv2.rectangle(bg,(128,48),(640-128,480-48),(0,255,0),3)
            # for i, top in enumerate(top_3):
            #     cv2.putText(bg, ges[top],(40,200-70*i), font, 1,(255,255,255),1)
            #     cv2.rectangle(bg,(400,225-70*i),(int(400+out[top]*170),205-70*i),(255,255,255),3)
        
            ret, buffer = cv2.imencode('.jpg', bg)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# TODO: handle multiple sets of labels (currently just Jester)
def plot_png():

    confidence_thresh = 0.6

    pos = range(len(gesture_labels))

    # create figure object, we don't use the matplotlib GUI 
    # so use the base figure class
    fig = Figure(figsize=(8,4))
    ax = fig.add_subplot(1, 1, 1)
    bars = ax.bar(pos, np.zeros(len(gesture_labels)), align="center")
    ax.set_ylim(0, 1)
    ax.set_xticks(pos)
    ax.set_xticklabels(gesture_labels, rotation=60, ha='right')
    ax.set_xlabel("Jester gesture classes")
    ax.set_ylabel("confidence")
    fig.tight_layout()

    while True:

        try:
            # read data from queue
            result = confidence_queue.get(timeout=0.2)

            # update the height for each bar
            for rect, y in zip(bars, result):
                if y > confidence_thresh:
                    rect.set_color("g")
                else:
                    rect.set_color("b")
                rect.set_height(y)

        except: # no data has been returned, detection is off
            pass
            # print("WARNING: no results returned")
     
        finally: 
            # write figure image to io buffer
            io_buffer = io.BytesIO()
            FigureCanvas(fig).print_png(io_buffer)
            io_buffer.seek(0)

            # pass bytes to webpage
            yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + io_buffer.read() + b'\r\n')


@app.route('/accuracy_plot')
def call_plot():
    return Response(plot_png(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Demo_Model_1_20BNJester_video_feed')
def Demo_Model_1_20BNJester_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Demo_Model_1_20BNJester_gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    plot_png()
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def holistic_model_gen(camera):
    """Video streaming generator function for holistic_model_gen."""

    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

                    # Flip the frame horizontally for a later selfie-view display, and convert
                    # the BGR frame to RGB.
                    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    # To improve performance, optionally mark the frame as not writeable to
                    # pass by reference.
                    frame.flags.writeable = False
                    results = holistic.process(frame)

                    # Draw landmark annotation on the image.
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles
                        .get_default_pose_landmarks_style())
               
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/holistic_model_video_feed')
def holistic_model_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(holistic_model_gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', threaded=True)
