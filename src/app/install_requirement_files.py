import os
import sys
import platform

PYTHONPATH = os.path.dirname(os.path.dirname(os.__file__))
print(PYTHONPATH)
print(platform.python_version())

# Where am I
Directory = os.path.dirname(sys._getframe().f_code.co_filename)

os.chdir(Directory)

python_path = sys.executable

if not os.path.isdir(Directory + '\\venv'):
   os.system(python_path + ' -m venv venv')

os.system('venv\\Scripts\\activate')  # <-- for Windows

os.system(python_path + ' -m pip install --upgrade pip')

req_files = [f for f in os.listdir(Directory) if f.endswith('.txt')]

# Look for and install all requirement files
for file in os.listdir(Directory):
   if file.startswith('requirement') and file.endswith('.txt'):
      os.system(python_path + ' -m pip install -r ' + file)

print("")
input('Installed Requirements. Press Enter to exit . . . ')
