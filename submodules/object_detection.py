import cv2
from ultralytics import YOLO
from PIL import Image

# Load the YOLO model
model = YOLO('model/object.pt')  # Adjust path as needed

# Function for YOLO Model Prediction (Object Detection)
def predict_from_cv2_frame(frame, threshold: float = 0.3):
    # Convert the frame (which is a NumPy array) to a PIL image for YOLO compatibility
    img_resized = cv2.resize(frame, (640, 640))  # Resize to 640x640 for YOLO
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_pil = Image.fromarray(img_resized)  # Convert to PIL image
    
    # Perform object detection on the image with the YOLO model
    results = model(img_pil, verbose=False)

    # Ensure results are not empty or undefined
    if not results or len(results) == 0 or results[0].boxes is None:
        return {"error": "Detection failed: No results found."}, 0

    # Get the bounding boxes, class indices, and confidence scores from the results
    boxes = results[0].boxes.xyxy  # Bounding boxes (xyxy)
    confidences = results[0].boxes.conf  # Confidence scores
    class_indices = results[0].boxes.cls  # Class indices

    # Filter out boxes with confidence below the threshold
    filtered_boxes = []
    for i, conf in enumerate(confidences):
        if conf >= threshold:
            box = boxes[i].tolist()  # Convert to list for easy processing
            class_name = model.names[int(class_indices[i])]  # Get the class name
            filtered_boxes.append({
                "class": class_name,
                "confidence": float(conf),
                "bbox": box
            })

    # Return the filtered annotations and the number of detections
    return filtered_boxes, len(filtered_boxes)
