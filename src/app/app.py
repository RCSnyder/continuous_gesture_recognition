from flask import Flask, render_template, Response, request
import requests
from importlib import import_module

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
import json
import time
import jsonify


label_dict = pd.read_csv('jester-v1-labels.csv', header=None)
ges = label_dict[0].tolist()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 48)

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
@app.route("/<any(Demo_Model_1_20BNJester, Model_2, Model_3):selected_model_name>")
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
        
        if not success:
            break
        else:
            
            ret, buffer = cv2.imencode('.jpg', frame)#bg)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def Demo_Model_1_20BNJester_gen(camera):
    """Video streaming generator function for Demo_Model_1_20BNJester."""
    fig, ax = plt.subplots()
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
    plt.ion()
    
    cooldown = 0
    eval_samples = 2
    num_classes = 27

    score_energy = torch.zeros((eval_samples, num_classes))

    while True:
        success, frame = camera.read()
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
                data = torch.cat(imgs).cuda()
                output = model(data.unsqueeze(0))
                out = (torch.nn.Softmax()(output).data).cpu().numpy()[0]
                if len(hist) > 300:
                    mean_hist  = mean_hist[1:]
                    hist  = hist[1:]
                out[-2:] = [0,0]
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
                    print('Gesture:', ges[indices], '\t\t\t\t\t\t Value: {:.2f}'.format(value.item()))
                    cooldown = 16 
                pred = indices
                imgs = imgs[1:]

                df = pd.DataFrame(mean_hist, columns=ges)

                ax.clear()
                df.plot.line(legend=False, figsize=(16,6),ax=ax, ylim=(0,1))
                if setup:
                    plt.show(block = False)
                    setup=False
                plt.draw()

            n += 1
            bg = np.full((480, 640, 3), 15, np.uint8)
            bg[:480, :640] = frame

            font = cv2.FONT_HERSHEY_SIMPLEX
            if value > 0.6:
                cv2.putText(bg, ges[pred],(10,10), font, 1,(0,0,0),2)
            cv2.rectangle(bg,(128,48),(640-128,480-48),(0,255,0),3)
            for i, top in enumerate(top_3):
                cv2.putText(bg, ges[top],(40,200-70*i), font, 1,(255,255,255),1)
                cv2.rectangle(bg,(400,225-70*i),(int(400+out[top]*170),205-70*i),(255,255,255),3)

        
            ret, buffer = cv2.imencode('.jpg', bg)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/Demo_Model_1_20BNJester_video_feed')
def Demo_Model_1_20BNJester_video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Demo_Model_1_20BNJester_gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(host='127.0.0.1', threaded=True)
