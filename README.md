# Video Image Extraction and Dataset Annotation Pipeline

## Overview
This project provides a robust and efficient pipeline to extract useful images from a video, annotate them using a general object detection model, and update an existing dataset. It is particularly optimized for selecting high-quality frames from videos based on specific thresholds for blur, motion, and shake, ensuring that only relevant frames with identifiable objects are processed.

## Features
1. **Efficient Frame Extraction**
   - The system uses a **two-pass approach** to determine and apply thresholds:
     - **Pass 1**: Analyzes the video to calculate thresholds for blur, motion, and shake.
     - **Pass 2**: Filters frames based on the calculated thresholds and checks for object presence.
   - Highly efficient in identifying frames that are sharp, stable, and contain objects of interest.

2. **Annotation**
   - Uses a general object detection model to annotate the extracted frames automatically.
   - Annotations are stored in YOLO format for seamless integration with machine learning workflows.

3. **Dataset Integration**
   - Updates an existing dataset by:
     - Adding new images and corresponding annotations.
     - Dynamically updating the dataset's YAML configuration.

4. **Data Upload**
   - Automatically splits the updated dataset into **training**, **validation**, and **test** sets.
   - Uploads the organized dataset to an **S3 bucket** for cloud storage and further use.

---

## How It Works

### Frame Extraction
1. **Two-Pass Filtering Process**:
   - **Pass 1**: 
     - Analyzes each frame for:
       - **Blur**: Measured using the Laplacian variance.
       - **Motion**: Determined by frame-to-frame differences.
       - **Shake**: Assessed based on displacement metrics.
     - Calculates thresholds for each metric based on the analysis.
   - **Pass 2**:
     - Filters frames using the calculated thresholds from Pass 1.
     - Retains only frames that:
       - Meet the quality criteria (sharp, stable).
       - Contain detectable objects.

2. **Advantages**:
   - Efficiently processes large videos.
   - Ensures only high-quality, relevant frames are retained for annotation.

### Annotation
- A general object detection model is applied to:
  - Detect objects in the extracted frames.
  - Generate YOLO format annotations for each frame.
- The annotations include bounding boxes and class labels for each detected object.

### Dataset Update
- Updates the existing dataset by:
  - Adding new images and their annotations.
  - Dynamically modifying the dataset's YAML file to reflect the new class distribution.

### Dataset Splitting and Uploading
- Splits the dataset into:
  - **Training** (70%)
  - **Validation** (20%)
  - **Test** (10%)
- Organizes the split dataset into separate directories.
- Uploads the organized dataset to an S3 bucket for cloud storage.



## Output Structure
The pipeline generates the following:
- Extracted frames (filtered and annotated).
- YOLO format annotations for each frame.
- Updated dataset split into `train`, `val`, and `test` directories.
- Uploaded dataset stored in the configured S3 bucket.
