import cv2
import torch
from ultralytics import YOLO
import pandas

model = YOLO("best.pt")

results = model("forward.png")

img = cv2.imread("forward.png")

df = results.pandas().xyxy[0]  

for _, row in df.iterrows():
    
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['name']  
    conf = row['confidence']  

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    text = f"{label} ({conf:.2f})"
    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("Results", img)
cv2.waitKey(0)
cv2.destroyAllWindows()