import cv2
import torch
import numpy as np
from ultralytics import YOLO

def mask_and_crop(image_path, model_path, target_class, save_cropped=True):
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Perform inference
    results = model(image)
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id == target_class:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Create mask
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                
                # Apply mask to image
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                
                # Crop detected section
                cropped_image = image[y1:y2, x1:x2]
                
                if save_cropped:
                    cropped_path = "cropped_image.jpg"
                    cv2.imwrite(cropped_path, cropped_image)
                    print(f"Cropped image saved as {cropped_path}")
                
                # Display the cropped image
                cv2.imshow("Cropped Image", cropped_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                return cropped_image
    
    print("Target class not found in image.")
    return None

# Example usage
image_path = "input.jpg"  # Replace with your image path
model_path = "yolov8_custom.pt"  # Replace with your trained YOLOv8 model path
target_class = 0  # Replace with the class ID to detect and crop

cropped_section = mask_and_crop(image_path, model_path, target_class)
