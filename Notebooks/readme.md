# 📊 Data Visualization for Object Detection Preprocessing

This section provides a detailed visual analysis of the dataset used for object detection using YOLO. The dataset is split into **training**, **testing**, and **validation** sets, each containing images and corresponding YOLO-style label files.

---

## 📄 Dataset Information

- **Project Name**: Table Detection
- **Format**: YOLOv5 / YOLOv8
- **Classes**:  
  - `0`: occupied
  - `1`: unoccupied_clean
  - `2`: unoccupied_dirty
- **Total Splits**:
  - `train/`: images and labels for training
  - `valid/`: images and labels for validation
  - `test/`: images and labels for testing

---

## 🧾 data.yaml

```yaml
train: train/images
val: valid/images
test: test/images

nc: 3
names: ['occupied,unoccupied_clean,unoccupied_dirty']

