# 3D-object-detection-using-stereo-images
This project implements a 3D object detection using stereo camera images. The system utilizes depth estimation from stereo pairs, detect objects and predicts their 3D bounding boxes. It combines state-of-the-art 2D object detection with specialized 3D bounding box regression, leveraging stereo depth estimation to bridge the gap between 2D and 3D understanding. The modular architecture allows for independent optimization of each component and easy integration of advances in individual stages.

## Features

- Stereo-based 3D Detection: Detects cars, pedestrians, and cyclists in 3D space using only stereo camera images
- YOLOv8-based 2D object detection
- 3D bounding box regression
- Visualization tools for 3D object detection
- Training framework with KITTI dataset support

### Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- tqdm
### Model Weights

The trained model weights are too large to include in this repository. 
You can download them from [Google Drive](https://drive.google.com/drive/folders/1Oepwc-61Vaxppkszfm1ZsJ0mH2hU2pFM?usp=drive_link).
### Installation

```bash
git clone https://github.com/yourusername/stereo-3d-object-detection.git
cd 3D-Object-Detection-using-Stereo-Images
pip install torch torchvision opencv-python numpy matplotlib ultralyticspip 
```
### Demo
Run the following command for demo and press 'enter' to view the depth map and 3D object detection of multiple demo images. 
```bash
python3 demo.py --left_img demo_imgs/left --right_img demo_imgs/right --model_weights model_weights/final_model.pth --display
```

## Model Architecture

The 3D object detection pipeline consists of 3 main components:

1. **Stereo Depth Estimation**: Computes depth maps from stereo image pairs
2. **2D Object Detection**: Uses YOLOv8 to detect objects in the scene
3. **3D Box Regression**: Predicts 3D bounding box parameters (position, dimensions, orientation)
