from ultralytics import YOLO
import numpy as np
import cv2

def crop_image(source_img, model_path, Taregt_cls , save_crop = True):
    model = YOLO(model_path)
    img = cv2.imread(source_img)
    # height, width = img.shape
    results = model(img)
    cropped_images_list = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls.item())
            
            if class_id == int(Taregt_cls):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img[y1:y2, x1:x2]
                cropped_images_list.append(cropped_img)
                cropped_path = f"Cropped image {len(cropped_images_list)}.jpg"
                if save_crop == True:
                    
                    cv2.imwrite(cropped_path,cropped_img)
    if cropped_images_list:
        for n, cropped_img in enumerate(cropped_images_list):
            cv2.imshow(f"Cropped Image {n+1}", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return cropped_images_list
    else:
        print("Target class nhi milli !")
        return None

def mirror(soruce_img_list):
    for soruce_img in soruce_img_list:
        img = cv2.imread(soruce_img)
        mirrored_img = cv2.flip(img,1)
    return mirrored_img
    pass
src_imgwa = "forward.png"
modelwa = "best.pt"

lala = crop_image(src_imgwa,modelwa,3,save_crop=False)
# print(lala)
