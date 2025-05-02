#!/usr/bin/env python
import os
import argparse
import cv2
import numpy as np
import time
from models import TrainedStereoObjectDetection3D
from detector_visualization import (
    colorize_depth_map, 
    draw_simple_cuboid, 
    visualize_with_cuboids,
    COCO_CLASSES
)

def run_demo():
    
    parser = argparse.ArgumentParser(description='3D Stereo Object Detection Demo with 3D Cuboids')
    parser.add_argument('--left_img', required=True, help='Path to left stereo image or directory')
    parser.add_argument('--right_img', required=True, help='Path to right stereo image or directory')
    parser.add_argument('--model_weights', required=True, help='Path to trained model weights')
    parser.add_argument('--yolo_model', default='yolov8s.pt', help='Path to YOLOv8 model')
    parser.add_argument('--output_dir', default='demo_output', help='Directory to save results')
    parser.add_argument('--display', action='store_true', help='Display results in window')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Initializing model with weights: {args.model_weights}")
    model = TrainedStereoObjectDetection3D(
        yolo_model=args.yolo_model,
        regressor_weights_path=args.model_weights
    )
    
    if os.path.isdir(args.left_img) and os.path.isdir(args.right_img):
        left_files = [f for f in sorted(os.listdir(args.left_img)) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        right_files = [f for f in sorted(os.listdir(args.right_img)) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        common_files = set(left_files).intersection(set(right_files))
        print(f"Found {len(common_files)} matching stereo pairs")
        
        for filename in sorted(common_files):
            left_path = os.path.join(args.left_img, filename)
            right_path = os.path.join(args.right_img, filename)
            process_image_pair(model, left_path, right_path, args.output_dir, args.display)
    else:
        process_image_pair(model, args.left_img, args.right_img, args.output_dir, args.display)
    
    print("Demo completed!")

def process_image_pair(model, left_path, right_path, output_dir, display=False):
    
    print(f"Processing: {os.path.basename(left_path)} & {os.path.basename(right_path)}")
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"Error: Could not load images")
        return
    
    start_time = time.time()
    results = model.detect(left_img, right_img)
    inference_time = time.time() - start_time
    
    num_detections = len(results['boxes_2d'])
    print(f"Detected {num_detections} objects in {inference_time:.3f} seconds")
    
    cuboid_vis = visualize_with_cuboids(left_img.copy(), results)
    depth_vis = colorize_depth_map(results['depth_map'])
    base_name = os.path.splitext(os.path.basename(left_path))[0]
    cuboid_path = os.path.join(output_dir, f"{base_name}_3d_cuboids.jpg")
    depth_path = os.path.join(output_dir, f"{base_name}_depth.jpg")
    
    cv2.imwrite(cuboid_path, cuboid_vis)
    cv2.imwrite(depth_path, depth_vis)
    print(f"Results saved to {cuboid_path} and {depth_path}")
    
    if display:
        max_height = 800
        if cuboid_vis.shape[0] > max_height:
            scale = max_height / cuboid_vis.shape[0]
            cuboid_vis = cv2.resize(cuboid_vis, None, fx=scale, fy=scale)
            depth_vis = cv2.resize(depth_vis, None, fx=scale, fy=scale)
        
        cv2.imshow("3D Object Detection", cuboid_vis)
        cv2.imshow("Depth Map", depth_vis)
        print("Press any key to continue...")
        cv2.waitKey(0)
    
    for i, (box2d, box3d, score, label) in enumerate(zip(
            results['boxes_2d'], results['boxes_3d'], 
            results['scores'], results['labels'])):
        
        class_name = COCO_CLASSES.get(label, f"Class {label}")
        print(f"Object {i+1}: {class_name}")
        print(f"  Confidence: {score:.3f}")
        print(f"  3D Position: [x={box3d[0]:.2f}, y={box3d[1]:.2f}, z={box3d[2]:.2f}m]")
        print(f"  3D Dimensions: [h={box3d[3]:.2f}m, w={box3d[4]:.2f}m, l={box3d[5]:.2f}m]")
        print(f"  Orientation: {box3d[6]:.2f} rad")
        print()

if __name__ == "__main__":
    run_demo()