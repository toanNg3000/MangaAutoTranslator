import cv2
import numpy as np
from PIL import Image


def split_image(image, ratio=2):
    # Determine the dimensions of the image
    image_width, image_height = image.size
    slice_height = ratio*image_width
    # Calculate the number of full slices and the remaining height
    num_full_slices = image_height // slice_height
    remaining_height = image_height % slice_height

    # Prepare a list to hold the sliced images
    sliced_images = []

    # Create slices from the original image
    for i in range(num_full_slices):
        top_boundary = i * slice_height
        bottom_boundary = (i + 1) * slice_height

        # Extract a slice from the image
        image_slice = image.crop((0, top_boundary, image_width, bottom_boundary))

        # Add the sliced image to the list
        sliced_images.append(image_slice)

    # Handle the last slice with remaining height
    if remaining_height > 0:
        top_boundary = num_full_slices * slice_height
        bottom_boundary = image_height
        image_slice = image.crop((0, top_boundary, image_width, bottom_boundary))
        sliced_images.append(image_slice)

    return sliced_images

def cumulative_height(images):
    return sum(height for _, height in images)

def merge_bounding_boxes(boxes, threshold=10):
    # Sort the bounding boxes by ymin values
    boxes.sort(key=lambda x: (list(x.values())[0])[1])
    
    if len(boxes) > 0:
        merged = [boxes[0]]  # Start with the first box
    else:
        merged = []
    for i in range(1, len(boxes)):
        key1, (xmin1, ymin1, xmax1, ymax1) = list(merged[-1].items())[0]
        key2, (xmin2, ymin2, xmax2, ymax2) = list(boxes[i].items())[0]

        # Check if the keys are different
        if key1 != key2:
            # Calculate the vertical distance between the boxes
            vertical_distance = ymin2 - ymax1
            
            # If the vertical distance is below the threshold, merge the boxes
            if vertical_distance <= threshold:
                merged[-1][key1] = (
                    min(xmin1, xmin2),
                    min(ymin1, ymin2),
                    max(xmax1, xmax2),
                    max(ymax1, ymax2)
                )
            else:
                # Otherwise, keep the boxes separate
                merged.append(boxes[i])
        else:
            # If the keys are the same, keep the boxes separate
            merged.append(boxes[i])
    final_boxes = []
    for item in merged:
        for v in item.values():
            final_boxes.append(tuple(v))
    return final_boxes

def set_image_dpi(image_np):
    # Convert NumPy array to PIL image
    im = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    # im = Image.open('/mnt/Storage/project/hf/txt-detect-demo/output/arenow.jpg')
    # Get image dimensions
    length_x, width_y = im.size
    if length_x < 100 or width_y < 100:
        # Calculate resizing factor
        # factor = min(5, float(1024.0 / length_x))
        factor = 4
    else:
        factor = 2
    
    # Calculate new size
    size = int(factor * length_x), int(factor * width_y)
    
    # Resize the image
    im_resized = im.resize(size, Image.LANCZOS)
    
    # Set DPI to 300 PPI (pixels per inch)
    dpi = (300, 300)
    im_resized.info['dpi'] = dpi
    
    # Convert PIL image back to NumPy array
    im_resized_np = cv2.cvtColor(np.array(im_resized), cv2.COLOR_RGB2BGR)
    
    return im_resized_np