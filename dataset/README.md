# 🗂️ TableVision – Custom Dataset Creation

This README documents the process of creating our **custom object detection dataset** for the TableVision project, where we classify restaurant tables as:

- **Occupied**
- **Unoccupied-Clean**
- **Unoccupied-Dirty**

---

## ⚡ Problem Faced

We needed a dataset to train a YOLOv8 model for detecting the above table statuses, but:

- No suitable pre-existing dataset was available in YOLOv8 format.
- The only dataset we found led to **overfitting** due to poor variety and annotation quality.

---

## 🛠️ How We Created Our Own Dataset

### 1. Image Collection
We collected images from multiple datasets and open sources, making sure to include diverse lighting, angles, and table setups.

---

### 2. Manual Labeling (Initial 40 Images)
We imported the images into [**Roboflow**](https://roboflow.com) and:

- Manually labeled the first 40 images.
- Created bounding boxes for every table.
- Labeled each table as either:
  - `occupied`
  - `unoccupied_clean`
  - `unoccupied_dirty`

---

### 3. Custom Model Training on Roboflow
Roboflow lets you train a model on your own labeled data. So we:

- Trained custom models using the 40 labeled images.
- Experimented with different subsets of 40 images.
- Selected the model with the **best accuracy**.

---

### 4. Auto-Labeling Remaining Images
Using the selected Roboflow-trained model, we:

- Auto-labeled the remaining images in our dataset.
- While this was much faster, accuracy was not perfect.

---

### 5. Manual Review and Correction
To ensure quality:

- We carefully **reviewed and corrected** the auto-labeled images.
- Verified that **no bounding boxes were missed** or mislabeled.

---

### 6. Exporting the Final Dataset
Once verified:

- We exported the dataset in **YOLOv8 format** from Roboflow.
- The dataset is now ready for training with the Ultralytics YOLOv8 framework.

---

## ✅ Summary

Despite not having a ready-to-use dataset, we built our own high-quality labeled dataset by combining:

- Manual annotation
- Smart auto-labeling
- Careful post-verification

This gave us a robust dataset tailored exactly to our needs for **table status detection** in real-world restaurant settings.

---

