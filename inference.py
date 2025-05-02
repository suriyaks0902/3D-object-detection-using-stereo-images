import os
import numpy as np
import cv2
import argparse
from models import TrainedStereoObjectDetection3D

def main():
    parser = argparse.ArgumentParser(description='3D Stereo Object Detection with Trained Regressor')
    parser.add_argument('--left_img', required=True, help='Path to left image')
    parser.add_argument('--right_img', required=True, help='Path to right image')
    parser.add_argument('--yolo_model', default='yolov8s.pt', help='YOLOv8 model name or path')
    parser.add_argument('--regressor_weights', required=True, help='Path to trained regressor weights')
    parser.add_argument('--output_dir', default='outputs_trained', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    model = TrainedStereoObjectDetection3D(
        yolo_model=args.yolo_model,
        regressor_weights_path=args.regressor_weights
    )
    left_img = cv2.imread(args.left_img)
    right_img = cv2.imread(args.right_img)
    
    if left_img is None or right_img is None:
        print(f"Error, couldn't load images: {args.left_img}, {args.right_img}")
        return
    
    results = model.detect(left_img, right_img)
    print(f"Detected {len(results['boxes_2d'])} objects in {results['inference_time']:.3f}s")
    
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.left_img))
        depth_map = results['depth_map']
        depth_vis = 255 * (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        vis_img = model.visualize_results(left_img, results, output_path)
        print(f"Visualization saved to {output_path}")
        cv2.imshow("3D Stereo Object Detection", vis_img)
        cv2.imshow("Depth Map", depth_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    class_names = model.detector.names
    for i, (box2d, box3d, score, label) in enumerate(zip(
            results['boxes_2d'], results['boxes_3d'], results['scores'], results['labels'])):
        
        class_name = class_names[label] if label in class_names else f"Class {label}"
        
        print(f"Object {i+1}:")
        print(f"  Class: {class_name} (confidence: {score:.3f})")
        print(f"  2D box: [x1={box2d[0]:.1f}, y1={box2d[1]:.1f}, x2={box2d[2]:.1f}, y2={box2d[3]:.1f}]")
        print(f"  3D box: [x={box3d[0]:.2f}, y={box3d[1]:.2f}, z={box3d[2]:.2f}, h={box3d[3]:.2f}, w={box3d[4]:.2f}, l={box3d[5]:.2f}, θ={box3d[6]:.2f}]")

if __name__ == "__main__":
    main()