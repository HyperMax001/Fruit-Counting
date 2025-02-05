import math

def euclidean_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2

    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2

    return math.sqrt((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2)

def check_matching(source_img1, source_img2, model_path, target_cls_to_ignore=[3], threshold=50):
    model = YOLO(model_path)
    img1, img2 = source_img1, source_img2  
    results1 = model(img1)
    results2 = model(img2)
    bounding_box_set = []

    for r1 in results1:
        for box1 in r1.boxes:
            class_id1 = int(box1.cls.item())
            if class_id1 not in target_cls_to_ignore:
                x11, y11, x21, y21 = map(int, box1.xyxy[0])
                new_box = (x11, y11, x21, y21)

                if not any(euclidean_distance(new_box, existing_box) < threshold for existing_box in bounding_box_set):
                    bounding_box_set.append(new_box)

    for r2 in results2:
        for box2 in r2.boxes:
            class_id2 = int(box2.cls.item())
            if class_id2 not in target_cls_to_ignore:
                x12, y12, x22, y22 = map(int, box2.xyxy[0])
                new_box = (x12, y12, x22, y22)

                if not any(euclidean_distance(new_box, existing_box) < threshold for existing_box in bounding_box_set):
                    bounding_box_set.append(new_box)

    return bounding_box_set
