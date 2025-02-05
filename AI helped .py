import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLO model with pre-trained weights
model = YOLO('best.pt')  # Ensure 'best.pt' is in your working directory

# Function to detect fruits in an image
def detect_fruits(image_path):
    image = cv2.imread(image_path)
    results = model(image)  # Run YOLO on image
    detections = []
    
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, class_id = box.cpu().numpy()
            detections.append({'bbox': (x1, y1, x2, y2), 'class': int(class_id)})
    
    return detections

# Function to match objects between front and back images
def match_fruits(front_detections, back_detections, threshold=50):
    unique_fruits = []
    
    for fruit in front_detections:
        matched = False
        x1_f, y1_f, x2_f, y2_f = fruit['bbox']
        
        for back_fruit in back_detections:
            x1_b, y1_b, x2_b, y2_b = back_fruit['bbox']
            
            # Compute center points
            center_f = ((x1_f + x2_f) / 2, (y1_f + y2_f) / 2)
            center_b = ((x1_b + x2_b) / 2, (y1_b + y2_b) / 2)
            
            # Compute Euclidean distance between centers
            distance = np.linalg.norm(np.array(center_f) - np.array(center_b))
            
            if distance < threshold:
                matched = True
                break
        
        if not matched:
            unique_fruits.append(fruit)
    
    return unique_fruits + back_detections

# Function to count fruits by class
def count_fruits(fruit_list):
    fruit_counts = {}
    
    for fruit in fruit_list:
        fruit_class = fruit['class']
        if fruit_class not in fruit_counts:
            fruit_counts[fruit_class] = 0
        fruit_counts[fruit_class] += 1
    
    return fruit_counts

# Main execution
front_image = 'front.jpg'  # Change this to your front image path
back_image = 'back.jpg'    # Change this to your back image path

front_detections = detect_fruits(front_image)
back_detections = detect_fruits(back_image)

unique_fruits = match_fruits(front_detections, back_detections)
fruit_counts = count_fruits(unique_fruits)

print("Final fruit count by color:", fruit_counts)
