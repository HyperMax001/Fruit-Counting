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