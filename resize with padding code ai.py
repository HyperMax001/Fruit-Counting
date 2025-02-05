import cv2
import numpy as np

def resize_with_padding(image_path, target_size=(640, 480)):
    image = cv2.imread(image_path)
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

# Example usage
output_image = resize_with_padding("cropped_image.jpg")
cv2.imwrite("padded_resized_image.jpg", output_image)
