import cv2
import torch
from ultralytics import YOLO
import pandas

model = YOLO("best.pt")
img = cv2.imread("cropped_forward_1.jpg")

results = model(img)




for r in results:
    for box in r.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])  # Class ID
        conf = box.conf[0]  # Confidence score
        label = model.names[class_id]  # Get class name

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class label & confidence
        text = f"{label} ({conf:.2f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()