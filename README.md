
# continuous_gesture_recognition, a live demo to evaluate and compare different models for Continuous Hand Gesture Recognition task performance

## Requirements (Preferred)

- Windows 10
- CUDA enabled graphics card
- Anaconda uninstalled
- Python >3.9

## Directions

- Download latest CUDA version from here:

```
https://developer.nvidia.com/cuda-downloads
```

- Ensure you have Python 3.9 installed or download it from here:

```
https://www.python.org/downloads/
```

- For simplicity sake, just put Python39 directory in C:/Python39

- Navigate to a preferred directory for the project and type the following commands:

```
git clone https://github.com/RCSnyder/continuous_gesture_recognition.git
```

### For Windows 10

- Navigate to ./src/app/
- Double click on install_requirement_files.py
- Double click on app.py to run the main program.

### If on Windows 10 and the script does not work perform the virtual environment creation and requirements.txt installation manually by following these steps

```
cd continuous_gesture_recognition\src\app
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio===0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
set FLASK_APP=app.py
flask run
```
