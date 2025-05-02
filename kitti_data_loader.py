import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import logging
from ultralytics import YOLO
from model_components import (
    KITTIStereoDepthEstimator, 
    ModifiedStereoBoundingBoxRegressor, 
    Box3DRegressionLoss, 
    box_iou, 
    calculate_3d_iou
)

logger = logging.getLogger(__name__)

class KITTI3DDataset(Dataset):

    def __init__(self, kitti_root, split='train', use_default_structure=True):
        
        self.kitti_root = kitti_root
        self.use_default_structure = use_default_structure
        split_file = os.path.join(kitti_root, 'split', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded {len(self.image_ids)} image IDs from {split_file}")
        self.setup_directories()
        self.depth_estimator = KITTIStereoDepthEstimator()
        self.yolo_model_path = 'yolov8s.pt'
        self._detector = None 
    
    @property
    def detector(self):
        
        if self._detector is None:
            self._detector = YOLO(self.yolo_model_path)
        return self._detector
        
    def setup_directories(self):
        
        structures = [
            {
                "name": "Nested structure",
                "left_img": os.path.join(self.kitti_root, "image_2", "training", "image_2"),
                "right_img": os.path.join(self.kitti_root, "image_3", "training", "image_3"),
                "calib": os.path.join(self.kitti_root, "calib", "training", "calib"),
                "label": os.path.join(self.kitti_root, "label_2", "training", "label_2")
            },
            {
                "name": "Flat structure",
                "left_img": os.path.join(self.kitti_root, "image_2"),
                "right_img": os.path.join(self.kitti_root, "image_3"),
                "calib": os.path.join(self.kitti_root, "calib"),
                "label": os.path.join(self.kitti_root, "label_2")
            },
            {
                "name": "Alternative structure",
                "left_img": os.path.join(self.kitti_root, "training", "image_2"),
                "right_img": os.path.join(self.kitti_root, "training", "image_3"),
                "calib": os.path.join(self.kitti_root, "training", "calib"),
                "label": os.path.join(self.kitti_root, "training", "label_2")
            }
        ]
        
        best_structure = None
        best_match_count = 0
        
        for structure in structures:
            match_count = 0
            for key, path in structure.items():
                if key != "name" and os.path.exists(path) and os.path.isdir(path):
                    match_count += 1
            
            if match_count > best_match_count:
                best_structure = structure
                best_match_count = match_count
        
        if best_match_count == 0:
            raise ValueError(f"Could not find a valid directory structure in {self.kitti_root}")
        
        logger.info(f"Using directory structure: {best_structure['name']}")
        
        self.left_img_dir = best_structure["left_img"]
        self.right_img_dir = best_structure["right_img"]
        self.calib_dir = best_structure["calib"]
        self.label_dir = best_structure["label"]
    
    def __len__(self):
        return len(self.image_ids)
    
    def parse_calib(self, calib_file):
        
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        calib_data = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                calib_data[key] = np.array([float(x) for x in value.split()])
            else:
                parts = line.split()
                if len(parts) > 1:
                    key = parts[0]
                    calib_data[key] = np.array([float(x) for x in parts[1:]])
        
        # projection matrices
        if 'P2' in calib_data:
            P2 = calib_data['P2'].reshape(3, 4)  # left camera projection matrix
        else:
            logger.warning(f"P2 not found in calibration file {calib_file}, using default")
            P2 = np.array([[721.5377, 0, 609.5593, 0], 
                          [0, 721.5377, 172.8540, 0], 
                          [0, 0, 1, 0]])
        
        if 'P3' in calib_data:
            P3 = calib_data['P3'].reshape(3, 4)  # right camera projection matrix
        else:
            logger.warning(f"P3 not found in calibration file {calib_file}, using default")
            P3 = np.array([[721.5377, 0, 609.5593, -388.3447], 
                          [0, 721.5377, 172.8540, 0], 
                          [0, 0, 1, 0]])
        
        # calibration parameters
        focal_length = P2[0, 0] 
        baseline = (P3[0, 3] - P2[0, 3]) / -P3[0, 0]  
        
        return {
            'P2': P2, 
            'P3': P3, 
            'focal_length': focal_length, 
            'baseline': baseline
        }
    
    def parse_label(self, label_file):
        
        objects = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 15:  
                    continue
                    
                if parts[0] in ['DontCare', 'Misc']: 
                    continue
                
                obj = {
                    'type': parts[0],
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                    'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                    'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                    'rotation_y': float(parts[14])
                }
                objects.append(obj)
        return objects
    
    def extract_features(self, left_img, bbox, depth_map):
        
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(left_img.shape[1]-1, x2), min(left_img.shape[0]-1, y2)
        
        if x2 <= x1 + 5 or y2 <= y1 + 5:
            return None
        
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
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        box_features = np.array([center_x, center_y, width, height])
        
        combined_features = np.concatenate([img_features, depth_features, box_features])
        return torch.tensor(combined_features, dtype=torch.float32)
    
    def match_detections_to_labels(self, detections, gt_objects, iou_threshold=0.5):
        
        matches = []
        
        for det_idx, det_box in enumerate(detections['boxes_2d']):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_obj in enumerate(gt_objects):
                gt_box = gt_obj['bbox']
                gt_box = [gt_box[0], gt_box[1], gt_box[2], gt_box[3]]
                iou = box_iou(det_box, gt_box)
                
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matches.append((det_idx, best_gt_idx, best_iou))
        
        return matches
    
    def __getitem__(self, idx):
        
        img_id = self.image_ids[idx]
        left_img_path = os.path.join(self.left_img_dir, f'{img_id}.png')
        right_img_path = os.path.join(self.right_img_dir, f'{img_id}.png')
        calib_path = os.path.join(self.calib_dir, f'{img_id}.txt')
        label_path = os.path.join(self.label_dir, f'{img_id}.txt')
        
        for file_path in [left_img_path, right_img_path, calib_path, label_path]:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None
        
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None or right_img is None:
            logger.warning(f"Error loading images for {img_id}")
            return None
        
        calib = self.parse_calib(calib_path)
        gt_objects = self.parse_label(label_path)
        self.depth_estimator.focal_length = calib['focal_length']
        self.depth_estimator.baseline = calib['baseline']
        depth_map = self.depth_estimator.compute_depth(left_img, right_img)
        
        try:
            with torch.no_grad():  
                results = self.detector(left_img)
        except Exception as e:
            logger.error(f"Error running YOLO detection: {e}")
            return None
        
        if len(results) == 0:
            return None
        
        boxes = []
        scores = []
        labels = []
        
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
            scores.append(box.conf[0].cpu().numpy())
            labels.append(int(box.cls[0].cpu().numpy()))
        
        detections = {
            'boxes_2d': np.array(boxes),
            'scores': np.array(scores),
            'labels': np.array(labels)
        }
        
        matches = self.match_detections_to_labels(detections, gt_objects)
        if len(matches) == 0:
            return None
        features_list = []
        targets_list = []
        
        for det_idx, gt_idx, _ in matches:
            features = self.extract_features(left_img, detections['boxes_2d'][det_idx], depth_map)
            
            if features is None:
                continue
            
            gt_obj = gt_objects[gt_idx]
            
            # gt 3D box parameters
            h, w, l = gt_obj['dimensions'] 
            x, y, z = gt_obj['location']    
            rotation_y = gt_obj['rotation_y']  
            
            # target vector
            target = torch.tensor([x, y, z, h, w, l, rotation_y], dtype=torch.float32)
            
            features_list.append(features)
            targets_list.append(target)
        
        if len(features_list) == 0:
            return None
        
        features_batch = torch.stack(features_list)
        targets_batch = torch.stack(targets_list)
        
        return features_batch, targets_batch
