import cv2

def decode_fourcc(cc):
    """
    transform int code into human readable str 
    https://stackoverflow.com/a/49138893/8615419
    """
    return "".join([chr((int(cc) >> 8 * i) & 0xFF) for i in range(4)])

def print_cap_props(capture: cv2.VideoCapture):

    props = {
        "CAP_PROP_FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
        "CAP_PROP_FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
        "CAP_PROP_FPS": cv2.CAP_PROP_FPS,
        "CAP_PROP_FOURCC": cv2.CAP_PROP_FOURCC,
        "CAP_PROP_BUFFERSIZE": cv2.CAP_PROP_BUFFERSIZE,
        "CAP_PROP_CONVERT_RGB": cv2.CAP_PROP_CONVERT_RGB,
        "CAP_PROP_AUTO_WB": cv2.CAP_PROP_AUTO_WB,
        "CAP_PROP_BACKEND": cv2.CAP_PROP_BACKEND,
        "CAP_PROP_CODEC_PIXEL_FORMAT": cv2.CAP_PROP_CODEC_PIXEL_FORMAT,
        # "CAP_PROP_HW_ACCELERATION": cv2.CAP_PROP_HW_ACCELERATION,
        # "CAP_PROP_HW_DEVICE": cv2.CAP_PROP_HW_DEVICE,
        # "CAP_PROP_HW_ACCELERATION_USE_OPENCL": cv2.CAP_PROP_HW_ACCELERATION_USE_OPENCL,
    }

    for key, val in props.items():
        if "FOURCC" in key:
            print(f"{key:>35} ={decode_fourcc(capture.get(val)):>6}")
        else:
            print(f"{key:>35} ={capture.get(val):6.0f}")
