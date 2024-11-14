By default, Python may include the user site-packages directory in sys.path. To prevent this:

Set the PYTHONNOUSERSITE Environment Variable:

export PYTHONNOUSERSITE=1

Run your script with this variable:
PYTHONNOUSERSITE=1 python your_script.py

echo "export PYTHONNOUSERSITE=1" >> ~/.bashrc
source ~/.bashrc
