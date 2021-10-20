# testing new backend

import ctypes
import io
import multiprocessing
import time
import logging
import queue

import cv2
import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import torch
import torchvision.transforms

from DemoModel import FullModel
# from utils.image_generator import imgGenerator
from ringbuffer import RingBuffer
from utils.fps import FPS

# define size of images to be stored in buffer
IMG_WIDTH = 640
IMG_HEIGHT = 480
IMG_CHANNELS = 3

# RECORDING_FREQ = 0.060 # ms; ~15 fps
RECORDING_FREQ = 0.030 # ms; ~30 fps

class Frame(ctypes.Structure):
    """c struct for representing frame and timestamp"""
    _fields_ = [
        ("timestamp_us", ctypes.c_ulonglong),
        ("frame", ctypes.c_ubyte * IMG_CHANNELS * IMG_WIDTH * IMG_HEIGHT)
    ]

def camera_proc(ring_buffer):

    cap = cv2.VideoCapture(0)

    # i = 0
    cam_fps = FPS()

    img = np.empty((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), dtype=np.uint8)

    while True:

        # see utils/image_generator
        # img = next(img_gen)
        time_s = time.time()

        while True:
            ret = cap.grab()
            if (time.time() - time_s) > RECORDING_FREQ:
                break

        ret, img = cap.retrieve()

        # resize to expected, convert to RGB
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not ret:
            break

        frame = Frame(int(time_s * 10e6), np.ctypeslib.as_ctypes(img))

        ring_buffer.write(frame)
        # logger.info(f"camera: {cam_fps():.2f} fps")

        # if i and i % 100 == 0:
        #     print('Wrote %d so far' % i)

        # i += 1
        # cv2.waitKey(1)


    ring_buffer.writer_done()
    logger.info('Writer is done')


def model_reader(ring_buffer, n, confidence_queue):
    
    logger.info("initializing model")
    t = time.time()

    model = FullModel(batch_size=1, seq_lenght=16)
    loaded_dict = torch.load('test_newbackend/demo.ckp')
    model.load_state_dict(loaded_dict)
    model = model.cuda()
    model.eval()

    # call model on dummy data to build graph
    model(torch.zeros((1, 16, 3, 96, 96), dtype=torch.float32).cuda())

    std, mean = [0.2674,  0.2676,  0.2648], [ 0.6,  0.6,  0.4]
    transform = torchvision.transforms.Compose([
        torchvision.transforms.transforms.Resize((120,160)),
        torchvision.transforms.transforms.CenterCrop((96, 96)),
        torchvision.transforms.transforms.Normalize(std=std, mean=mean),
    ])

    model_fps = FPS()

    logger.info(f"initialization finished, elapsed time: {time.time() - t:.2f} seconds")    
    logger.info("beginning detection")
    while True:
                
        try:
            data = ring_buffer.blocking_read(None, length=n)
        except ring_buffer.WriterFinishedError:
            break
        
        # structured array
        tp = np.dtype(Frame)
        arr = np.frombuffer(data, dtype=tp)

        # accessing structured numpy array
        timestamps = arr["timestamp_us"]
        frames = arr["frame"]
        # print(frames.shape) # (16, 480, 640, 3)
        
        # format array as expected by torch
        # float tensor: [0,1]
        # expected shape = (frames, channels, height, width)
        frames = frames.transpose(0,3,1,2) / 255.
        # print(frames.shape) # (16, 3, 480, 640)
        
        # convert np.array to torch.tensor
        frame_tensor = torch.Tensor(frames)

        # preprocess frames
        imgs = transform(frame_tensor).cuda()
        # print(imgs.shape) # (16, 3, 96, 96)
        # print(imgs.dtype)

        # predict on frames (after adding a batch dim) 
        output = model(imgs.unsqueeze(dim=0))

        # model output
        confidences = (torch.nn.Softmax(dim=1)(output).data).cpu().numpy()[0]

        # logger.info(f"model : {model_fps():.2f} predictions / second")

        # put confidences in output queue
        while True:
            try:
                confidence_queue.put_nowait(confidences)
                break
            except queue.Full:
                continue
        
    # print('Reader %r is done' % id(pointer))


def display_reader(ring_buffer, n: int=1):

    cv2.namedWindow("test")

    while True:

        try:
            data = ring_buffer.blocking_read(None, length=n)
        except ring_buffer.WriterFinishedError:
            break

        # structured array
        tp = np.dtype(Frame)
        arr = np.frombuffer(data, dtype=tp)

        # accessing structured array 
        timestamps = arr["timestamp_us"]
        frames = arr["frame"]

        # print(f"Reader saw records at timestamp {timestamps[0]} to {timestamps[1]}, frame_shape={frames.shape}")

        cv2.imshow("test", cv2.flip(cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR), 1))
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # print('Reader %r is done' % id(pointer))

with open("src/app/jester-v1-labels.txt", "r") as fh:
    gesture_labels = fh.read().splitlines()

def plot_png(confidence_queue):

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
            confidences = confidence_queue.get_nowait()

            # update the height for each bar
            for rect, y in zip(bars, confidences):
                if y > confidence_thresh:
                    rect.set_color("g")
                else:
                    rect.set_color("b")
                rect.set_height(y)

        except queue.Empty: # no data has been returned, detection is off
            continue
            # print("WARNING: no results returned")

        finally:
            # hacky, already implemented for web UI
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()

            X = np.asarray(buf) #.transpose()
            # print(X.shape)

            cv2.imshow("test_plot", cv2.cvtColor(X, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)            
            
    cv2.destroyAllWindows()


if __name__=="__main__":
    logger = multiprocessing.log_to_stderr(logging.INFO)

    # define ring buffer large enough to hold N "frames" (ctypes.Structure defined above)
    ring_buffer = RingBuffer(c_type=Frame, slot_count=32)

    # define queue to pass confidence data to plotting process
    confidence_queue = multiprocessing.Queue(maxsize=2)
    
    ring_buffer.new_writer()
    ring_buffer.new_reader()
    ring_buffer.new_reader()

    processes = [
        multiprocessing.Process(target=model_reader, args=(ring_buffer, 16, confidence_queue, )),
        multiprocessing.Process(target=display_reader, args=(ring_buffer, 1, )),
        multiprocessing.Process(target=camera_proc, args=(ring_buffer, )),
        multiprocessing.Process(target=plot_png, args=(confidence_queue, ))
    ]

    for p in processes:
        p.daemon = True
        p.start()

    # only run for 2 minutes
    time.sleep(120)

    for p in processes:
        p.terminate()
    