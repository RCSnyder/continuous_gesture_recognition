
# continuous_gesture_recognition, a live demo to evaluate and compare different models for Continuous Hand Gesture Recognition task performance

## Directions

- Type the following commands:

```
git clone https://github.com/RCSnyder/continuous_gesture_recognition.git
```

- Ensure you have Python 3.9 installed or download it from here https://www.python.org/downloads/
- For simplicity sake, have Python39 directory in C:/Python39

### For Windows 10

- Navigate to ./src/app/
- Double click on install_requirement_files.py
- Double click on app.py to run the main program.

### If on Linux, perform the virtual environment creation and requirements.txt installation manually

```
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ./src/app/
set FLASK_APP=app.py
flask run
```

### If on Windows and the script does not work perform the virtual environment creation and requirements.txt installation manually by following these steps

```
python -m venv env
env/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ./src/app/
set FLASK_APP=app.py
flask run
```
