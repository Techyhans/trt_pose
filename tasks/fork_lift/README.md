# Dependency
1. Install OpenCV. Refer [this link](https://automaticaddison.com/how-to-install-opencv-4-5-on-nvidia-jetson-nano/) to install opencv in jetson nano 
   
2. Install PyTorch and Torchvision.  To do this on NVIDIA Jetson, we recommend following [this guide](https://forums.developer.nvidia.com/t/72048)

2. Install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

    ```python
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt
    cd torch2trt
    sudo python3 setup.py install --plugins
    ```

3. Install other miscellaneous packages

    ```python
    sudo pip3 install tqdm cython pycocotools
    sudo apt-get install python3-matplotlib
    ```
    
### Step 2 - Install trt_pose

```python
git clone https://github.com/NVIDIA-AI-IOT/trt_pose
cd trt_pose
sudo python3 setup.py install
```

### Step 3 - Forklift

variable : 
1. model file name 
2. topology file name - default: forklift.json
3. camera : "csi" or 0 for csi camera or usb camera
```
cd tasks/fork_lift
python3 detect.py <<model file name>> forklift.json <<camera>> 
```