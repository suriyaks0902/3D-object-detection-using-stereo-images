# 3D-object-detection-using-stereo-images
---

## 🧭 Overview
This project implements a stereo-vision-based 3D object detection system for autonomous driving. It integrates YOLOv8 2D detection, stereo depth estimation, and a 3D bounding box regression network to localize objects in 3D space using only camera inputs. Trained on the KITTI dataset, the model achieves real-time performance (120 ms/frame) and demonstrates strong coarse 3D localization, offering a cost-efficient, LiDAR-free alternative for real-time perception.

> **Goal:** Build a cost-efficient, LiDAR-free perception system using stereo cameras while maintaining real-time 3D understanding.

---

## 🧱 System Architecture

**Pipeline Components:**
1. **Stereo Depth Estimation** – Generates depth maps using Semi-Global Matching (SGM).  
2. **2D Object Detection** – Detects objects from the left stereo image using YOLOv8.  
3. **Feature Extraction** – Combines RGB and depth features within detected bounding boxes.  
4. **3D Box Regression** – Predicts 3D bounding box parameters `(x, y, z, h, w, l, θ)` using a 5-layer MLP.

<p align="center">
  <img src="https://github.com/user-attachments/assets/38208599-195f-4a91-945b-70c965ce3d65" alt="3D Object Detection Pipeline" width="500"/>
</p>

---

## 🧮 Mathematical Model

Depth estimation from stereo disparity:
\[
\text{depth} = \frac{f \times b}{d}
\]
Where:  
- \( f \): focal length  
- \( b \): baseline distance between stereo cameras  
- \( d \): disparity between matched pixel pairs

**Loss Function:**
\[
L = \alpha L_{pos} + \beta L_{dim} + \gamma L_{rot}
\]

| Term | Description |
|------|--------------|
| \(L_{pos}\) | MSE on position (x, y, z) |
| \(L_{dim}\) | MSE on dimensions (h, w, l) |
| \(L_{rot}\) | \(1 - \cos(\theta_{pred} - \theta_{gt})\) |

---

## ⚙️ Implementation Details

- **Language:** Python 3.9  
- **Frameworks:** PyTorch, OpenCV, Ultralytics YOLOv8  
- **Dataset:** [KITTI 3D Object Detection Benchmark](http://www.cvlibs.net/datasets/kitti/)  
- **Training:** Adam optimizer (lr=1e-4), batch size=8  
- **Augmentations:** Random flips, bounding-box jittering  

---

## 🧩 Qualitative Results

### **Urban Driving Scenes**

<p align="center">
  <img src="https://github.com/user-attachments/assets/5a68cfc8-ca95-4cd3-bf06-92e779f35f22" width="45%">
  <img src="https://github.com/user-attachments/assets/1923896b-9a7c-481a-800e-39f583311cc7" width="45%"><br>
  <img src="https://github.com/user-attachments/assets/b722ef04-657b-49b9-b49f-e321aaf7f140" width="45%">
  <img src="https://github.com/user-attachments/assets/f4c386e6-d853-40a8-a870-77561f015831" width="45%">
</p>

*Figure 2: Qualitative examples of 3D bounding box detection in **urban driving scenes**.*


---

### **Rural Driving Scenes**

<p align="center">
  <img src="https://github.com/user-attachments/assets/20a6f951-67ec-460c-8f61-31f36c7d9a6d" width="45%">
  <img src="https://github.com/user-attachments/assets/4efcd903-2559-4a6e-a481-25d9c64ef283" width="45%"><br>
  <img src="https://github.com/user-attachments/assets/3c042c20-f4e7-47e1-97a6-b571d57dde5c" width="45%">
  <img src="https://github.com/user-attachments/assets/3ab0a274-9481-4663-a84a-8fe9aa6d5a86" width="45%">
</p>

*Figure 3: Qualitative examples of 3D bounding box detection in rural driving scenes.*

---

### **Training and Validation Performance**

<p align="center">
  <img src="https://github.com/user-attachments/assets/b24aeb0e-408c-4757-9acf-953cc8395df5" width="1000" alt="Training Curves">
</p>

*Figure 4: Training and validation loss curves and validation 3D IoU.*

---

## 📊 Quantitative Results

| Metric | Value |
|---------|-------|
| Average 3D IoU | 0.297 |
| AP@0.25 | 55.6% |
| AP@0.50 | 22.2% |
| AP@0.70 | 3.4% |
| Mean Position Error | 1.13 m |
| Mean Dimension Error | 0.517 m |
| Mean Rotation Error | 83.6° |
| Inference Time | 120 ms |

**Performance Highlights:**
- Achieved **real-time speed** (120ms per frame) suitable for onboard deployment.  
- Depth features significantly improve 3D localization (drop from 55.6% → 42.3% AP@0.25 when removed).  
- Compared with DISP R-CNN: faster inference (4× speedup) but lower accuracy at fine alignment.

---

## 🔬 Ablation Study

| Variant | AP@0.25 | AP@0.50 | Position Error (m) | Runtime (ms) |
|----------|----------|----------|--------------------|---------------|
| Full System | 55.6 | 22.2 | 1.13 | 120 |
| w/o Depth Features | 42.3 | 16.4 | 1.48 | 115 |
| w/o Image Features | 37.8 | 13.2 | 1.70 | 90 |
| w/ PSMNet Depth | 63.2 | 31.5 | 0.96 | 470 |
| w/ Basic Regressor | 51.7 | 19.5 | 1.28 | 110 |

---

## 🧠 Key Insights

- Combining **depth and RGB features** enhances 3D bounding box accuracy.  
- **Stereo block matching** provides a robust trade-off between accuracy and real-time performance.  
- **Orientation prediction** remains the main challenge — future work will explore multi-bin and quaternion-based representations.

---

## 🚀 Future Work

- **Improve orientation estimation** using angular discretization and road priors.  
- **Hybrid depth estimation**: integrate deep stereo (e.g., PSMNet) with SGM.  
- **3D-aware feature extraction** and attention-based multi-scale representations.  
- **Deploy on embedded hardware** (e.g., NVIDIA Jetson) for on-vehicle inference.

---

## 📚 References

Key papers referenced:
- [Stereo R-CNN for 3D Object Detection](https://arxiv.org/abs/1902.09738) — Li et al., 2019  
- [PSMNet: Pyramid Stereo Matching Network](https://arxiv.org/abs/1803.08669) — Chang & Chen, 2018  
- [PointPillars](https://arxiv.org/abs/1812.05784) — Lang et al., 2019  
- [VoxelNet](https://arxiv.org/abs/1711.06396) — Zhou & Tuzel, 2018  
- [DSGN](https://arxiv.org/abs/2001.03398) — Chen et al., 2020  

Full list of citations available in the project report.

---
