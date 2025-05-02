#!/usr/bin/env python
import os
import argparse
import cv2
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
from models import TrainedStereoObjectDetection3D

COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

def draw_simple_cuboid(img, box2d, box3d, class_name, score, color=(0, 255, 0)):
    
    x1, y1, x2, y2 = map(int, box2d)
    depth = box3d[2]  
    x, y, z, h, w, l, theta = box3d
    width = x2 - x1
    height = y2 - y1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    perspective_factor = min(50.0 / (depth + 0.1), 0.3) 
    shift_x = int(width * perspective_factor)
    shift_y = int(height * perspective_factor)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    front_top_left = (x1, y1)
    front_top_right = (x2, y1)
    front_bottom_left = (x1, y2)
    front_bottom_right = (x2, y2)
    
    back_top_left = (
        int(x1 + shift_x + l*perspective_factor*cos_theta), 
        int(y1 - shift_y - l*perspective_factor*sin_theta)
    )
    back_top_right = (
        int(x2 + shift_x + l*perspective_factor*cos_theta), 
        int(y1 - shift_y - l*perspective_factor*sin_theta)
    )
    back_bottom_left = (
        int(x1 + shift_x + l*perspective_factor*cos_theta), 
        int(y2 - shift_y - l*perspective_factor*sin_theta)
    )
    back_bottom_right = (
        int(x2 + shift_x + l*perspective_factor*cos_theta), 
        int(y2 - shift_y - l*perspective_factor*sin_theta)
    )
    cv2.rectangle(img, front_top_left, front_bottom_right, color, 1)
    cv2.rectangle(img, back_top_left, back_bottom_right, color, 1)
    cv2.line(img, front_top_left, back_top_left, color, 1)
    cv2.line(img, front_top_right, back_top_right, color, 1)
    cv2.line(img, front_bottom_left, back_bottom_left, color, 1)
    cv2.line(img, front_bottom_right, back_bottom_right, color, 1)
    
    front_center = (int((front_top_left[0] + front_bottom_right[0]) / 2), 
                    int((front_top_left[1] + front_bottom_right[1]) / 2))
    back_center = (int((back_top_left[0] + back_bottom_right[0]) / 2), 
                   int((back_top_left[1] + back_bottom_right[1]) / 2))
    cv2.line(img, front_center, back_center, (0, 0, 255), 2)  
    label_text = f"{class_name}: {score:.2f}"
    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # depth 
    depth_text = f"Depth: {depth:.2f}m"
    cv2.putText(img, depth_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 3d dimensions
    dim_text = f"HxWxL: {h:.2f}x{w:.2f}x{l:.2f}m"
    cv2.putText(img, dim_text, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # rotation angle
    rot_text = f"theta: {theta:.2f}rad"
    cv2.putText(img, rot_text, (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img

def colorize_depth_map(depth_map, min_depth=0, max_depth=50):
    
    depth_map = np.clip(depth_map, min_depth, max_depth)
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth)
    normalized_depth = (normalized_depth * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    
    return colored_depth

def visualize_with_cuboids(image, results, output_path=None):
    
    vis_img = image.copy()
    for i, (box2d, box3d, score, label) in enumerate(zip(
            results['boxes_2d'], results['boxes_3d'], 
            results['scores'], results['labels'])):
        class_name = COCO_CLASSES.get(label, f"Class {label}")
        cuboid_color = (0, 0, 255) if class_name in ['car', 'truck', 'bus', 'bicycle', 'motorcycle'] else (0, 255, 0)
        vis_img = draw_simple_cuboid(vis_img, box2d, box3d, class_name, score, color=cuboid_color)
    
    combined_vis = vis_img
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, combined_vis)
        print(f"Visualization saved to {output_path}")
    
    return combined_vis

def process_image_pair(model, left_path, right_path, output_dir, display=False, verbose=True):
    
    os.makedirs(output_dir, exist_ok=True)
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"Error: Could not load images: {left_path}, {right_path}")
        return None
    
    base_name = os.path.splitext(os.path.basename(left_path))[0]
    start_time = time.time()
    results = model.detect(left_img, right_img)
    processing_time = time.time() - start_time
    
    if verbose:
        print(f"Processing image pair: {base_name}")
        print(f"Image dimensions: {left_img.shape[1]}x{left_img.shape[0]}")
        print(f"Detection completed in {processing_time:.3f}s")
        print(f"Detected {len(results['boxes_2d'])} objects")
    
    vis_path = os.path.join(output_dir, f"{base_name}_cuboids.jpg")
    vis_img = visualize_with_cuboids(left_img, results, vis_path)
    
    if verbose and len(results['boxes_2d']) > 0:
        print("\nDetection Results:")
        
        for i, (box2d, box3d, score, label) in enumerate(zip(
                results['boxes_2d'], results['boxes_3d'], 
                results['scores'], results['labels'])):
            
            class_name = COCO_CLASSES.get(label, f"Class {label}")
            
            print(f"Object {i+1}:")
            print(f"  Class: {class_name} (confidence: {score:.3f})")
            print(f"  3D position: [x={box3d[0]:.2f}, y={box3d[1]:.2f}, z={box3d[2]:.2f}m]")
            print(f"  3D dimensions: [h={box3d[3]:.2f}m, w={box3d[4]:.2f}m, l={box3d[5]:.2f}m, θ={box3d[6]:.2f}]")
    
    if display:
        window_name = "3D Object Detection with Trained Regressor"
        cv2.imshow(window_name, vis_img)
        depth_map = results['depth_map']
        depth_vis = colorize_depth_map(depth_map)
        cv2.imshow("Depth Map", depth_vis)
        print("Press any key to continue")
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
    return results

def run_demo():
    
    parser = argparse.ArgumentParser(description='3D Object Detection Visualization with Trained Regressor')
    parser.add_argument('--left_img', required=True, help='Path to left image or directory of images')
    parser.add_argument('--right_img', required=True, help='Path to right image or directory of images')
    parser.add_argument('--output_dir', default='demo_output_trained', help='Output directory')
    parser.add_argument('--yolo_model', default='yolov8s.pt', help='YOLOv8 model path')
    parser.add_argument('--regressor_weights', required=True, help='Path to trained regressor weights')
    parser.add_argument('--display', action='store_true', help='Display visualizations')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    model = TrainedStereoObjectDetection3D(
        yolo_model=args.yolo_model,
        regressor_weights_path=args.regressor_weights
    )
    
    if os.path.isdir(args.left_img) and os.path.isdir(args.right_img):
        left_files = sorted([f for f in os.listdir(args.left_img) if f.endswith(('.png', '.jpg', '.jpeg'))])
        right_files = sorted([f for f in os.listdir(args.right_img) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
        common_files = []
        for f in left_files:
            if f in right_files:
                common_files.append(f)
        
        print(f"Found {len(common_files)} matching image pairs")
        
        for filename in common_files:
            left_path = os.path.join(args.left_img, filename)
            right_path = os.path.join(args.right_img, filename)
            print(f"\nProcessing: {filename}")
            process_image_pair(model, left_path, right_path, args.output_dir, args.display, not args.quiet)
    else:
        process_image_pair(model, args.left_img, args.right_img, args.output_dir, args.display, not args.quiet)
    
    print("\nDemo completed.")

if __name__ == "__main__":
    run_demo()