from ultralytics import YOLO
model = YOLO("best.pt")
results = model('forward.png', show = True)

df = results.pandas().xyxy[0]
for _, row in df.iterrows():
    # Extract bounding box coordinates and class label
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['name']  # Class name
    conf = row['confidence']  # Confidence score
print(label)
