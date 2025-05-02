import torch
import torch.nn as nn
import numpy as np
import cv2

class KITTIStereoDepthEstimator:
    
    def __init__(self, num_disparities=128, block_size=11):
        self.num_disparities = num_disparities
        self.block_size = block_size
        self.focal_length = 721.5377  
        self.baseline = 0.54  
    
    def compute_depth(self, left_img, right_img):
        
        stereo = cv2.StereoBM_create(numDisparities=self.num_disparities, blockSize=self.block_size)
        stereo.setMinDisparity(0)
        stereo.setUniquenessRatio(5)
        stereo.setSpeckleWindowSize(100)
        stereo.setSpeckleRange(32)
        stereo.setDisp12MaxDiff(1)

        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            right_gray = right_img
        
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity[disparity == 0] = 0.1
        depth = self.focal_length * self.baseline / disparity
        depth = np.clip(depth, 0, 100)
        
        return depth

class ModifiedStereoBoundingBoxRegressor(nn.Module):
    
    def __init__(self, input_dim=12288+5+4, output_dim=7):
        super(ModifiedStereoBoundingBoxRegressor, self).__init__()
        
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

class Box3DRegressionLoss(nn.Module):
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5):
        super(Box3DRegressionLoss, self).__init__()
        self.alpha = alpha  # weight for position error
        self.beta = beta    # weight for dimension error
        self.gamma = gamma  # weight for rotation error
    
    def forward(self, pred, target):
        
        pos_error = torch.mean(torch.sum((pred[:, :3] - target[:, :3]) ** 2, dim=1))
        dim_error = torch.mean(torch.sum((pred[:, 3:6] - target[:, 3:6]) ** 2, dim=1))
        rot_pred = pred[:, 6]
        rot_target = target[:, 6]
        rot_error = torch.mean(1 - torch.cos(rot_pred - rot_target))
        total_loss = self.alpha * pos_error + self.beta * dim_error + self.gamma * rot_error
        
        return total_loss

def box_iou(box1, box2):
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
    
    return intersection_area / union_area

def calculate_3d_iou(box1, box2):
    
    x1, y1, z1, h1, w1, l1, _ = box1
    x2, y2, z2, h2, w2, l2, _ = box2
    
    box1_min = [x1 - l1/2, y1 - h1/2, z1 - w1/2]
    box1_max = [x1 + l1/2, y1 + h1/2, z1 + w1/2]
    
    box2_min = [x2 - l2/2, y2 - h2/2, z2 - w2/2]
    box2_max = [x2 + l2/2, y2 + h2/2, z2 + w2/2]
    
    intersection_min = [max(box1_min[0], box2_min[0]),
                       max(box1_min[1], box2_min[1]),
                       max(box1_min[2], box2_min[2])]
    
    intersection_max = [min(box1_max[0], box2_max[0]),
                       min(box1_max[1], box2_max[1]),
                       min(box1_max[2], box2_max[2])]
    
    if (intersection_min[0] >= intersection_max[0] or
        intersection_min[1] >= intersection_max[1] or
        intersection_min[2] >= intersection_max[2]):
        return 0.0
    
    intersection_vol = ((intersection_max[0] - intersection_min[0]) *
                       (intersection_max[1] - intersection_min[1]) *
                       (intersection_max[2] - intersection_min[2]))
    
    box1_vol = h1 * w1 * l1
    box2_vol = h2 * w2 * l2
    
    union_vol = box1_vol + box2_vol - intersection_vol
    
    return intersection_vol / union_vol