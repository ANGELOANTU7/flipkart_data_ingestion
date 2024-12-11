import cv2
import os
from submodules.object_detection import predict_from_cv2_frame
from submodules.video_analyzer import VideoAnalyzer
from typing import Dict
import numpy as np
import datetime
import uuid
import random
from typing import Dict, Tuple
import yaml
import requests
import boto3
from collections import OrderedDict


def adjust_thresholds_cli(analyzer: VideoAnalyzer) -> Dict:
    """Command-line interface for adjusting thresholds"""
    while True:
        # Calculate current thresholds
        thresholds = analyzer._calculate_thresholds()
        
        analyzer.config.save_config()
        print("Settings saved!")
        return thresholds
        



def process_video(
    input_path: str, 
    output_folder: str, 
    thresholds: Dict, 
    class_name: str = "object", 
    s3_bucket: str = "flipkarttest", 
    s3_key: str = "data.yaml", 
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)
):
    """
    Process the video, save frames with YOLO v1.0 annotations, and split into train, test, validation sets.
    Updates class names and number in the YAML file stored in S3 if necessary.
    """
    # Step 1: Download YAML file from S3
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name=os.getenv("REGION_NAME")
    )
    yaml_path = os.path.join("/tmp", "temp_classes.yaml")
    s3.download_file(s3_bucket, s3_key, yaml_path)

    # Step 2: Parse YAML file
    with open(yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    if "names" not in yaml_data or "nc" not in yaml_data:
        raise ValueError("YAML file is missing required keys: 'names' and 'nc'")

    # Step 3: Check and update class names and count
    if class_name not in yaml_data["names"]:
        yaml_data["names"].append(class_name)
        yaml_data["nc"] = len(yaml_data["names"])

        # Step 4: Prepare the updated YAML data in the correct format
        formatted_yaml = f"""train: {yaml_data.get('train', '../train/images')}
    val: {yaml_data.get('val', '../valid/images')}
    test: {yaml_data.get('test', '../test/images')}
    nc: {yaml_data['nc']}
    names: {yaml_data['names']}"""

        # Save the updated YAML file locally first
        with open(yaml_path, "w") as file:
            # Strip any potential leading/trailing whitespace from each line
            cleaned_yaml = "\n".join(line.strip() for line in formatted_yaml.splitlines())
            file.write(cleaned_yaml)

        # Upload the updated YAML file back to S3
        s3.upload_file(yaml_path, s3_bucket, s3_key)

    # Get updated class information
    class_id = yaml_data["names"].index(class_name)
    nc = yaml_data["nc"]

    # Clean up temporary file
    os.remove(yaml_path)

    # Proceed with video processing as before
    images_folder = os.path.join(output_folder, "temp_images")
    labels_folder = os.path.join(output_folder, "temp_labels")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    saved_frames = 0

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read first frame")

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"Processing frame {processed_frames}/{total_frames}")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        frame_delta = cv2.absdiff(prev_frame_gray, gray_frame_blur)
        threshold_frame = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

        contours, _ = cv2.findContours(threshold_frame.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        movement_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < thresholds['movement_threshold']:
                continue
            movement_detected = True
            break

        if movement_detected:
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            if laplacian_var < thresholds['blur_threshold']:
                continue

            diff = cv2.absdiff(prev_frame_gray, gray_frame_blur)
            non_zero_count = np.count_nonzero(diff)
            total_pixels = diff.size
            shake_metric = (non_zero_count / total_pixels) * 100

            if shake_metric > thresholds['shake_threshold']:
                continue
            
            
            # Apply object detection
            annotations, num_objects = predict_from_cv2_frame(frame)

            # Save the frame and annotations
            unique_id = uuid.uuid4().hex  # Generate a globally unique identifier
            image_path = os.path.join(images_folder, f"frame_{unique_id}.jpg")
            annotation_path = os.path.join(labels_folder, f"frame_{unique_id}.txt")

            cv2.imwrite(image_path, frame)
            saved_frames += 1

            # Write YOLO annotations
            img_height, img_width, _ = frame.shape
            with open(annotation_path, "w") as f:
                for annotation in annotations:
                    x_min, y_min, x_max, y_max = annotation["bbox"]
                    
                    # Normalize bbox coordinates
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    bbox_width = (x_max - x_min) / img_width
                    bbox_height = (y_max - y_min) / img_height
                    
                    # Write in YOLO format
                    f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

        prev_frame_gray = gray_frame_blur

    cap.release()

    # Call modified split_data function with s3 client
    split_data(images_folder, labels_folder, output_folder, split_ratios, s3, s3_bucket)
    print(f"\nProcessing complete!")
    print(f"Processed {processed_frames} frames")
    print(f"Saved {saved_frames} frames with annotations")



def split_data(
    images_folder: str,
    labels_folder: str,
    output_folder: str,
    split_ratios: Tuple[float, float, float],
    s3_client,
    s3_bucket: str
):
    """
    Split the dataset into train, test, and validation sets and upload to S3.

    Args:
        images_folder (str): Path to the folder containing images.
        labels_folder (str): Path to the folder containing labels.
        output_folder (str): Temporary path for processing.
        split_ratios (Tuple[float, float, float]): Ratios for train, test, and validation splits.
        s3_client: Boto3 S3 client instance.
        s3_bucket (str): Name of the S3 bucket.
    """
    import shutil  # Add this import at the top of your file
    
    image_files = sorted(os.listdir(images_folder))
    total_files = len(image_files)
    indices = list(range(total_files))
    random.shuffle(indices)

    # Calculate split indices
    train_end = int(total_files * split_ratios[0])
    test_end = train_end + int(total_files * split_ratios[1])

    splits = {
        "train": indices[:train_end],
        "test": indices[train_end:test_end],
        "valid": indices[test_end:]
    }

    for split_name, split_indices in splits.items():
        # Create temporary local directories
        split_images_folder = os.path.join(output_folder, split_name, "images")
        split_labels_folder = os.path.join(output_folder, split_name, "labels")
        
        os.makedirs(split_images_folder, exist_ok=True)
        os.makedirs(split_labels_folder, exist_ok=True)

        for index in split_indices:
            base_name = image_files[index].split(".")[0]
            
            # Local paths
            image_src = os.path.join(images_folder, f"{base_name}.jpg")
            label_src = os.path.join(labels_folder, f"{base_name}.txt")
            image_tmp = os.path.join(split_images_folder, f"{base_name}.jpg")
            label_tmp = os.path.join(split_labels_folder, f"{base_name}.txt")

            # Move files to temporary location
            os.rename(image_src, image_tmp)
            os.rename(label_src, label_tmp)

            # Upload to S3
            s3_image_key = f"{split_name}/images/{base_name}.jpg"
            s3_label_key = f"{split_name}/labels/{base_name}.txt"
            
            try:
                # Upload image and label to S3
                s3_client.upload_file(image_tmp, s3_bucket, s3_image_key)
                s3_client.upload_file(label_tmp, s3_bucket, s3_label_key)
                
                # Remove temporary files after successful upload
                os.remove(image_tmp)
                os.remove(label_tmp)
            except Exception as e:
                print(f"Error uploading files for {base_name}: {str(e)}")
                continue

    # Cleanup all temporary directories safely
    try:
        # Clean up split directories
        split_dirs = [os.path.join(output_folder, split) for split in splits.keys()]
        for split_dir in split_dirs:
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)

        # Clean up original images and labels folders
        if os.path.exists(images_folder):
            shutil.rmtree(images_folder)
        if os.path.exists(labels_folder):
            shutil.rmtree(labels_folder)
            
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")

    print(f"Successfully uploaded split datasets to S3 bucket: {s3_bucket}")