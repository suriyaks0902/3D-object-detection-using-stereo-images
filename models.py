import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import time
from PIL import Image
from ultralytics import YOLO
from base_detector import StereoDepthEstimator

class StereoBoundingBoxRegressor(nn.Module):
    def __init__(self, input_dim=12288+5+4, output_dim=7):
        super(StereoBoundingBoxRegressor, self).__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim) 
        )
    
    def forward(self, features):
        return self.regressor(features)

class TrainedStereoObjectDetection3D:
    def __init__(self, yolo_model='yolov8s.pt', regressor_weights_path=None):
        
        try:
            print(f"Loading YOLOv8 model: {yolo_model}")
            self.detector = YOLO(yolo_model)
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise
        
        self.depth_estimator = StereoDepthEstimator(num_disparities=128, block_size=11)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if regressor_weights_path and os.path.exists(regressor_weights_path):
            checkpoint = torch.load(regressor_weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                first_layer_key = 'regressor.0.weight'
                if first_layer_key in checkpoint['model_state_dict']:
                    input_dim = checkpoint['model_state_dict'][first_layer_key].shape[1]
                    print(f"Detected input dimension: {input_dim}")
                else:
                    input_dim = 12288+5+4
                    print(f"Using default input dimension: {input_dim}")
            else:
                input_dim = 12288+5+4
                print(f"Using default input dimension: {input_dim}")
        else:
            input_dim = 12288+5+4
            print(f"Using default input dimension: {input_dim}")
        
        self.box3d_regressor = StereoBoundingBoxRegressor(input_dim=input_dim)
        self.box3d_regressor.to(self.device)
        
        if regressor_weights_path and os.path.exists(regressor_weights_path):
            print(f"Loading regressor weights from {regressor_weights_path}")
            checkpoint = torch.load(regressor_weights_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.box3d_regressor.load_state_dict(checkpoint['model_state_dict'])
            else:
                print("Error: model_state_dict not found in checkpoint")
                
            if 'val_iou' in checkpoint:
                print(f"Model validation IoU: {checkpoint['val_iou']:.4f}")
        else:
            print("Warning: No regressor weights provided or file not found")
        self.box3d_regressor.eval()
    
    def extract_features(self, left_img, bbox, depth_map):
        
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(left_img.shape[1]-1, x2), min(left_img.shape[0]-1, y2)
        
        img_patch = left_img[y1:y2, x1:x2]
        if img_patch.size == 0:  
            return None
        
        img_patch = cv2.resize(img_patch, (64, 64))
        depth_patch = depth_map[y1:y2, x1:x2]
        if depth_patch.size == 0:
            return None
        
        valid_depths = depth_patch[(depth_patch > 0) & (depth_patch < 100)]
        if len(valid_depths) == 0:
            return None
        
        mean_depth = np.mean(valid_depths)
        median_depth = np.median(valid_depths)
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        std_depth = np.std(valid_depths)
        
        img_patch = img_patch / 255.0
        img_patch = (img_patch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_features = img_patch.transpose(2, 0, 1).flatten()  
        depth_features = np.array([mean_depth, median_depth, min_depth, max_depth, std_depth])
        
        # box center and size features
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        box_features = np.array([center_x, center_y, width, height])

        combined_features = np.concatenate([img_features, depth_features, box_features])
        return torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)
    
    def detect(self, left_img, right_img):
        
        start_time = time.time()
        
        if isinstance(left_img, Image.Image):
            left_img = np.array(left_img)
            right_img = np.array(right_img)
        
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
                features = self.extract_features(left_img, [x1, y1, x2, y2], depth_map)
                
                if features is not None:
                    with torch.no_grad():
                        features = features.to(self.device)
                        box3d_pred = self.box3d_regressor(features).cpu().numpy().squeeze()
                        x, y, z, h, w, l, rot_y = box3d_pred
                        boxes_3d.append([x, y, z, h, w, l, rot_y])
                else:
                    roi_depth = depth_map[int(y1):int(y2), int(x1):int(x2)]
                    valid_depths = roi_depth[(roi_depth > 0) & (roi_depth < 100)]
                    if len(valid_depths) > 0:
                        mean_depth = np.mean(valid_depths)
                    else:
                        mean_depth = 20.0
                    
                    if labels[-1] == 0:  # Person
                        h, w, l = 1.8, 0.8, 0.8
                    elif labels[-1] == 2:  # Car
                        h, w, l = 1.5, 1.7, 4.5
                    elif labels[-1] == 7:  # Truck
                        h, w, l = 2.5, 2.5, 6.0
                    else:  # Default
                        h, w, l = 1.5, 1.5, 1.5
                    
                    # 3D center
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    cz = mean_depth
                    
                    # 3D bounding box 
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
        
    def visualize_results(self, left_img, results, output_path=None):
       
        img_vis = left_img.copy()
        
        for i, (box, score, label) in enumerate(zip(results['boxes_2d'], results['scores'], results['labels'])):
            x1, y1, x2, y2 = map(int, box)
            class_names = self.detector.names  
            class_name = class_names[label] if label in class_names else f"Class {label}"
            
            color = (0, 255, 0) 
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            text = f"{class_name}: {score:.2f}"
            cv2.putText(img_vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if i < len(results['boxes_3d']):
                box3d = results['boxes_3d'][i]
                x, y, z, h, w, l, theta = box3d
                cv2.circle(img_vis, (int(x), int(y)), 5, (0, 0, 255), -1)
                depth_text = f"Depth: {z:.2f}m"
                cv2.putText(img_vis, depth_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                dim_text = f"3D dims: {h:.2f}x{w:.2f}x{l:.2f}m"
                rot_text = f"Rotation: {theta:.2f}rad"
                cv2.putText(img_vis, dim_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(img_vis, rot_text, (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                line_length = 50
                end_x = int(x + line_length * cos_theta)
                end_y = int(y - line_length * sin_theta)  
                cv2.line(img_vis, (int(x), int(y)), (end_x, end_y), (0, 255, 255), 2)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, img_vis)
            print(f"Visualization saved to {output_path}")
        
        return img_vis