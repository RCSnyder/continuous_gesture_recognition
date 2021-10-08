
# Continuous gesture recognition
![Python Version](https://img.shields.io/badge/python-3.9-blue) ![Platform](https://img.shields.io/badge/platform-windows%7Clinux-lightgray)

A live demo to evaluate and compare different models for Continuous Hand Gesture Recognition task performance

## Requirements (Preferred)
- Windows 10
- CUDA enabled graphics card
- Anaconda uninstalled
- Python >3.9

## Build

- Ensure you have Python 3.9 installed or download it from here: https://www.python.org/downloads/

clone the repository
```
# Navigate to a preferred directory with >10gb space.
git clone https://github.com/RCSnyder/continuous_gesture_recognition.git
```

### Windows 10
Copy and paste this code into the terminal

```bash
cd continuous_gesture_recognition\src\app
python -m venv env
env\Scripts\activate
pip install --upgrade pip
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
set FLASK_APP=app.py
flask run
```

Go to 127.0.0.1:5000 in a web browser to view the app.


### Linux (with ROCm)
> tested on debian bullseye, python v3.9.2, ROCm stack v4.3.0

TODO: investigate pytorch rocm docker container.

```bash
cd continuous_gesture_recognition

# create and activate virtual environment
python -m venv .env
source .env/bin/activate

# upgrade pip
pip install --upgrade pip

# install pytorch: these may change depending on your install strategy. See Pytorch ROCm below.
pip install ~/Documents/projects/pytorch/dist/torch-1.11.0a0+git6559604-cp39-cp39-linux_x86_64.whl
pip install ~/Documents/projects/vision/dist/torchvision-0.10.0a0+e828eef-cp39-cp39-linux_x86_64.whl

# install other dependencies
pip install -r requirements.txt

# run app.
cd src/app
FLASK_APP=app.py flask run
```

### Drivers

**CUDA Drivers (NVIDIA GPU)**
- Download latest CUDA version from here: https://developer.nvidia.com/cuda-downloads


**ROCm (AMD GPU)**
- install rocm: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html
  - > ROCm may require dependency hacking for unsupported distros, see: https://github.com/RadeonOpenCompute/ROCm/issues/1125#issuecomment-719656556
  - > Not all hardware is supported, see: https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support

**Pytorch ROCm**
- pytorch-rocm wheel available: https://pytorch.org/get-started/locally/
  - > **WARNING**: using this resulted in model building errors on tested machine, but is quick and easy. Likely due to wheel being built for ROCm stack v4.2.
```bash
# as of 8 Oct, 2021
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/rocm4.2/torch_nightly.html
```

- building pytorch (rocm) from source
  - > this will take a several hours
```bash
# clone pytorch upstream, managed by ROCm (AMD devs)
git clone git@github.com:ROCmSoftwarePlatform/pytorch.git
cd pytorch
git submodule update --init --recursive

# create and activate virtual env
python3 -m venv .env
source .env/bin/activate

# install dependencies
pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# HIPify cuda (gpu lang) code
python tools/amd_build/build_amd.py

# build custom pytorch wheel, found in: pytorch/dist/*.whl
PYTORCH_ROCM_ARCH=<gfx_arch> MAX_JOBS=<n> python setup.py bdist_wheel
```
> PYTORCH_ROCM_ARCH = `$ rocminfo` will return all agents (if rocm is correctly installed). The "Name" listed for your gpu will start with "gfx". Default is multiarch, but if you don't have multiple gpu's this only adds extra compile time.

> MAX_JOBS = `(RAM in GB) / 4` as general rule of thumb, if you run into errors try decreasing

- building torchvision (rocm) from source
  - > use the same virtual env as the pytorch build
```bash
# clone torchvision upstream
git clone git@github.com:ROCmSoftwarePlatform/vision.git
cd vision
PYTORCH_ROCM_ARCH=<gfx_arch> MAX_JOBS=<n> python setup.py bdist_wheel
```

