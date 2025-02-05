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
                if save_crop:
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
    mirrored_imgs_list = []
    for soruce_img in soruce_img_list:
        img = cv2.imread(soruce_img)
        mirrored_img =[cv2.flip(img, 1) for img in soruce_img_list]
        mirrored_imgs_list.append(mirrored_img)
    return mirrored_imgs_list

def check_matching(source_img1,source_img2, model_path, Target_cls_to_ignore=3):
    model = YOLO(model_path)
    img1 = cv2.imread(source_img1)
    img2 = cv2.imread(source_img2)
    # height, width = img.shape
    results1 = model(img1)
    results2 = model(img2)
    bounding_box_set = set()

    for r1,r2 in zip(results1,results2):
        for box1, box2 in zip(r1.boxes,r2.boxes):
            class_id1 = int(box1.cls.item())
            class_id2 = int(box2.cls.item())
            if class_id1 != int(Target_cls_to_ignore):
                x11, y11, x21, y21 = map(int, box1.xyxy[0])
                bounding_box_list_of_points1 = [x11,y11,x21,y21]
                bounding_box_set.add(tuple(bounding_box_list_of_points1))
            if class_id2 != int(Target_cls_to_ignore):
                x12, y12, x22, y22 = map(int, box2.xyxy[0])
                bounding_box_list_of_points2 = [x12,y12,x22,y22]
                bounding_box_set.add(tuple(bounding_box_list_of_points2))

                        

            # if class_id2 != int(Target_cls_to_ignore):
            #     x12, y12, x22, y22 = map(int, box2.xyxy[0])
            

    return bounding_box_set

src_imgwa_fwd = "forward.jpg"
modelwa = "best.pt"
src_imgwa_bck = "backward.jpg"


forward_images = crop_image(src_imgwa_fwd,modelwa,3,save_crop=False)
backward_images = crop_image(src_imgwa_bck,modelwa,3,save_crop=False)
backward_mirrored_images = mirror(backward_images)

# reversed_backward_images_list = backward_images[::-1]
reversed_backward_mirrored_images_list = backward_mirrored_images[::-1]
count_of_fruit_on_plants = []
for image1,image2 in zip(forward_images,reversed_backward_mirrored_images_list):
    fruit_on_plant_set = check_matching(image1,image2,modelwa)
    count_of_fruit_on_plants.append(len(fruit_on_plant_set))
total_number_of_fruits = sum(x for x in count_of_fruit_on_plants if isinstance(x, int))
print(total_number_of_fruits)

# print(lala)
