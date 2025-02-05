from ultralytics import YOLO
import numpy as np
import cv2

def resize_without_scaling(image, target_size=(640, 480), fill_color=(0, 0, 0)):
    """
    Resize an image without scaling the main object. The cropped object remains the same size, 
    and the rest of the image is filled with a solid color.

    :param image: The cropped image (small object).
    :param target_size: The final output size (width, height).
    :param fill_color: The color to fill the empty space (default: black).
    :return: The resized image with padding.
    """
    h, w = image.shape[:2]
    
    # Create a blank image with the target size and fill it with the chosen color
    result = np.full((target_size[1], target_size[0], 3), fill_color, dtype=np.uint8)

    # Compute the placement position (centered)
    x_offset = (target_size[0] - w) // 2
    y_offset = (target_size[1] - h) // 2

    # Place the original image in the center without resizing it
    result[y_offset:y_offset+h, x_offset:x_offset+w] = image

    return result

def resize_with_padding(image, target_size=(640, 480)):
    # image = cv2.imread(image_path)
    h, w = image.shape[:2]
    
    # Compute scale factor while maintaining aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create new blank image (black background)
    result = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Compute padding
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2

    # Place resized image in the center
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return result

def crop_image(source_img, model_path, target_cls, save_crop=True):
    model = YOLO(model_path)
    img = cv2.imread(source_img)
    results = model(img)
    cropped_images_list = []
    cropped_images_paths_list = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls.item())
            
            if class_id == int(target_cls):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img[y1:y2, x1:x2]
                cropped_img_resized = resize_without_scaling(cropped_img)
                cropped_images_list.append(cropped_img_resized)
                
                if save_crop:
                    cropped_path = f"cropped_{source_img.split('.')[0]}_{len(cropped_images_list)}.jpg"
                    cropped_images_paths_list.append(cropped_path)
                    cv2.imwrite(cropped_path, cropped_img_resized)
    
    # if cropped_images_list:
    #     for n, cropped_img in enumerate(cropped_images_list):
    #         cv2.imshow(f"Cropped Image {n+1}", cropped_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    return cropped_images_list if cropped_images_list else None

def mirror(source_img_list):
    return [cv2.flip(img, 1) for img in source_img_list]

def check_matching(source_img1, source_img2, model_path, target_cls_to_ignore=3):
    model = YOLO(model_path)
    img1, img2 = source_img1, source_img2  
    results1 = model(img1)
    results2 = model(img2)
    bounding_box_set = set()
    
    for r1, r2 in zip(results1, results2):
        for box1, box2 in zip(r1.boxes, r2.boxes):
            class_id1 = int(box1.cls.item())
            class_id2 = int(box2.cls.item())
            if class_id1 !=target_cls_to_ignore:
                x11, y11, x21, y21 = map(int, box1.xyxy[0])
                bounding_box_set.add((x11, y11, x21, y21))
                print((x11, y11, x21, y21))

            if class_id2 != target_cls_to_ignore:
                x12, y12, x22, y22 = map(int, box2.xyxy[0])
                bounding_box_set.add((x12, y12, x22, y22))
                print((x12, y12, x22, y22))
    return bounding_box_set  # Fixed return value

def colour_of_fruit(source_img1, source_img2, model_path):
    model = YOLO(model_path)
    img1, img2 = source_img1, source_img2  
    results1 = model(img1)
    results2 = model(img2)

    class_id_list = []
    for r1, r2 in zip(results1, results2):
        for box1, box2 in zip(r1.boxes, r2.boxes):
            class_id1 = int(box1.cls.item())
            class_id2 = int(box2.cls.item())
            class_id_list.extend([class_id1, class_id2])  # Fixed append issue

    if 1 in class_id_list:
        return "Yellow Fruit"
    elif 2 in class_id_list:
        return "Purple Fruit"
    return "Unknown"  # Added default return value

src_img_fwd = "forward.jpg"
src_img_bck = "backward.jpg"
model_path = "best.pt"

forward_images = crop_image(src_img_fwd, model_path, 3, save_crop=True)
backward_images = crop_image(src_img_bck, model_path, 3, save_crop=True)

if forward_images and backward_images:
    backward_mirrored_images = mirror(backward_images)
    reversed_backward_mirrored_images = backward_mirrored_images[::-1]
    
    fruit_colour_set = set()
    count_of_fruit_on_plants = []
    
    for image1, image2 in zip(forward_images, reversed_backward_mirrored_images):
        fruit_on_plant_set = check_matching(image1, image2, model_path)
        count_of_fruit_on_plants.append(len(fruit_on_plant_set))
        
        fruit_colour = colour_of_fruit(image1, image2, model_path)
        fruit_colour_set.add(fruit_colour)  
    
    total_number_of_fruits = sum(x for x in count_of_fruit_on_plants if isinstance(x, int))
    
    print("Total number of fruits:", total_number_of_fruits)
    print(f"Fruit Type: {list(fruit_colour_set)}") 
else:
    print("Life Jhand h !! ")
