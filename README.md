# Air-mouse--gesture-control
Control your mouse pointer with natural hand gestures trained on Single Shot Detector built on MobileNet.

### Outcome: 
  * Keep pinched for moving the mouse around
  * Lift index finger for single click
  * Lift index and middle finger together for double click
<br>
<img src="https://github.com/wolf-hash/Air-mouse--gesture-control/blob/main/screenshots/airmouse.gif" width="850" height="400" />
 
 *Loss over the Training period*
<img src="https://github.com/wolf-hash/Air-mouse--gesture-control/blob/main/screenshots/Loss.jpg" width="500" height="400" />
 

## Installation: 
```
git clone https://github.com/wolf-hash/Air-mouse--gesture-control.git
cd Air-mouse--gesture-control
```

### Install Preliminaries
   #### Object Detection API (Required):
```
    mkdir API
    cd API
    git clone https://github.com/tensorflow/models.git
    cd models/research
    protoc object_detection/protos/*.proto --python_out=.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install --use-feature=2020-resolver .
```

   #### Install model of choice from Tensorflow Zoo (Only for training):
     Go to https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
     Download model of choice
     Single Shot Detector of MobileNet(320x320) is given in the pre-trained directory, however one can experiment on different models.
     Extract it to Air-mouse--gesture-control/pre-trained
     
  #### Install LabelImg for labelling custom dataset (Only for training):
  **Link** - https://github.com/tzutalin/labelImg
 ```
    git clone https://github.com/tzutalin/labelImg
 ```
   **For Linux**: 
``` 
    sudo apt-get install pyqt5-dev-tools
    sudo pip3 install -r requirements/requirements-linux-python3.txt
    make qt5py3
    python3 labelImg.py
```

### Run from pretrained model:
```
    conda create -n airmouse pip python=3.7.6
    conda activate airmouse
    pip install -r requirements.txt
    
    python main.py
```
