import cv2
import numpy as np
from submodules.threshold_config import ThresholdConfig
from typing import Dict


class VideoAnalyzer:
    def __init__(self, video_path: str, config: ThresholdConfig):
        self.video_path = video_path
        self.config = config
        self.laplacian_values = []
        self.movement_areas = []
        self.shake_metrics = []
        
    def analyze_video(self) -> Dict:
        """Analyze video to determine baseline metrics"""
        print("Analyzing video to determine baseline metrics...")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file")

        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Error: Could not read first frame")
        
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, (21, 21), 0)

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Analyzing frame {frame_count}/{total_frames}")

            # Calculate metrics
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            self.laplacian_values.append(laplacian_var)

            gray_frame_blur = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            frame_delta = cv2.absdiff(prev_frame_gray, gray_frame_blur)
            threshold_frame = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)
            
            contours, _ = cv2.findContours(threshold_frame.copy(), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                max_area = max(cv2.contourArea(c) for c in contours)
                self.movement_areas.append(max_area)

            diff = cv2.absdiff(prev_frame_gray, gray_frame_blur)
            non_zero_count = np.count_nonzero(diff)
            total_pixels = diff.size
            shake_metric = (non_zero_count / total_pixels) * 100
            self.shake_metrics.append(shake_metric)

            prev_frame_gray = gray_frame_blur

        cap.release()
        return self._calculate_thresholds()

    def _calculate_thresholds(self) -> Dict:
        """Calculate thresholds using adjustment factors"""
        def remove_outliers(data):
            if not data:
                return []
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return [x for x in data if lower_bound <= x <= upper_bound]

        clean_laplacian = remove_outliers(self.laplacian_values)
        clean_movement = remove_outliers(self.movement_areas)
        clean_shake = remove_outliers(self.shake_metrics)
        

        return {
            'blur_threshold': np.mean(clean_laplacian) * self.config.adjustment_factors['blur_factor'],
            'movement_threshold': np.mean(clean_movement) * self.config.adjustment_factors['movement_factor'],
            'shake_threshold': np.percentile(clean_shake, 95) * self.config.adjustment_factors['shake_factor']
        }