import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time
from ultralytics import YOLO  

class StereoDepthEstimator:
    def __init__(self, num_disparities=128, block_size=11):
        self.stereo = cv2.StereoBM_create(numDisparities=num_disparities,blockSize=block_size)
        self.stereo.setMinDisparity(0)
        self.stereo.setUniquenessRatio(5)
        self.stereo.setSpeckleWindowSize(100)
        self.stereo.setSpeckleRange(32)
        self.stereo.setDisp12MaxDiff(1)      
        self.focal_length = 721.5377  
        self.baseline = 0.54 
    
    def compute_depth(self, left_img, right_img):

        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity[disparity == 0] = 0.1
        depth = self.focal_length * self.baseline / disparity
        depth = np.clip(depth, 0, 100)
        
        return depth

class StereoBoundingBoxRegressor(nn.Module):

    def __init__(self, input_dim=512, output_dim=7):
        super(StereoBoundingBoxRegressor, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim) 
        )
    
    def forward(self, features):
        return self.regressor(features)

class StereoObjectDetection3D:
    
    def __init__(self, yolo_model='yolov8s.pt'):
        
        try:
            print(f"loading YOLOv8 model: {yolo_model}")
            self.detector = YOLO(yolo_model)
        except Exception as e:
            print(f"error loading YOLOv8 model: {e}")
            raise
        
        self.depth_estimator = StereoDepthEstimator(num_disparities=128, block_size=11)
        self.box3d_regressor = StereoBoundingBoxRegressor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"device:{self.device}")
        self.box3d_regressor.to(self.device)
        self.image_size = (384, 1248) 
    
    def detect(self, left_img, right_img):
        
        start_time = time.time()
        
        if isinstance(left_img, Image.Image):
            left_img = np.array(left_img)
            right_img = np.array(right_img)
        left_img_vis = left_img.copy()
        if left_img.shape[2] == 4:  
            left_img = left_img[:, :, :3]
            right_img = right_img[:, :, :3]
        depth_map = self.depth_estimator.compute_depth(left_img, right_img)
        results = self.detector(left_img)
        
        boxes = []
        scores = []
        labels = []
        boxes_3d = []
        
        if len(results) > 0:
            result = results[0]
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
                boxes.append([x1, y1, x2, y2])
                scores.append(box.conf[0].cpu().numpy())  
                labels.append(int(box.cls[0].cpu().numpy())) 
                roi_depth = depth_map[int(y1):int(y2), int(x1):int(x2)]
                
                valid_depths = roi_depth[(roi_depth > 0) & (roi_depth < 100)]
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                else:
                    mean_depth = 20.0
                if labels[-1] == 0:  
                    h, w, l = 1.5, 1.7, 4.5
                elif labels[-1] == 1:  
                    h, w, l = 1.8, 0.8, 0.8
                else:  
                    h, w, l = 1.5, 1.5, 1.5
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                cz = mean_depth
                
                box3d = [cx, cy, cz, h, w, l, 0.0]  
                boxes_3d.append(box3d)
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels)
        boxes_3d = np.array(boxes_3d)
        inference_time = time.time() - start_time
        
        results = {
            'boxes_2d': boxes,
            'boxes_3d': boxes_3d,
            'scores': scores,
            'labels': labels,
            'depth_map': depth_map,
            'inference_time': inference_time
        }
        
        return results
