from mltranslator.modules.inpainting.lama import LaMa
from mltranslator.modules.inpainting.schema import Config
from mltranslator.utils.detection import combine_results, make_bubble_mask, bubble_interior_bounds
from mltranslator.utils.textblock import TextBlock, sort_regions, visualize_textblocks
from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
from mltranslator import PROJECT_DIR
from PIL import Image
import datetime

def generate_mask(img: np.ndarray, blk_list: List[TextBlock], default_kernel_size=5):
    h, w, c = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # Start with a black mask

    for blk in blk_list:
        seg = blk.segm_pts
        # if blk.source_lang == 'en':
        #     default_kernel_size = 1
        kernel_size = default_kernel_size # Default kernel size
        if blk.text_class == 'text_bubble':
            # Access the bounding box coordinates
            bbox = blk.bubble_xyxy
            # Calculate the minimal distance from the mask to the bounding box edges
            min_distance_to_bbox = min(
                np.min(seg[:, 0]) - bbox[0],  # left side
                bbox[2] - np.max(seg[:, 0]),  # right side
                np.min(seg[:, 1]) - bbox[1],  # top side
                bbox[3] - np.max(seg[:, 1])   # bottom side
            )
            # adjust kernel size if necessary
            if default_kernel_size >= min_distance_to_bbox:
                kernel_size = max(5, int(min_distance_to_bbox-(0.2*min_distance_to_bbox)))

        # Create a kernel for dilation based on the kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Draw the individual mask and dilate it
        single_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(single_mask, [seg], 255)
        single_mask = cv2.dilate(single_mask, kernel, iterations=3)

        # Merge the dilated mask with the global mask
        mask = cv2.bitwise_or(mask, single_mask)
        np.expand_dims(mask, axis=-1)

    return mask

class Inpaintor:
    def __init__(self):
        self.bubble_detection = YOLO(f'{PROJECT_DIR}/mltranslator/models/detection/best.pt')
        self.text_segmentation = YOLO(f'{PROJECT_DIR}/mltranslator/models/inpainting/comic-text-segmenter.pt')
        img_size_process=512
        self.inpainter = LaMa('cuda')
        self.conf = Config(hd_strategy="Crop", hd_strategy_crop_trigger_size=img_size_process,
                           hd_strategy_resize_limit=img_size_process, hd_strategy_crop_margin=2)

    def inpaint(self, pil_img):
        yolo_device='cuda'
        text_detec_result = self.bubble_detection(pil_img, device=yolo_device, half=True, imgsz=640, conf=0.5, verbose=False)[0] 
        txt_seg_result = self.text_segmentation(pil_img, device=yolo_device, half=True, imgsz=1024, conf=0.1, verbose=False)[0]

        combined = combine_results(text_detec_result, txt_seg_result)

        blk_list: List[TextBlock] = []
        for txt_bbox, bble_bbox, txt_seg_points, txt_class in combined:
            text_region = TextBlock(txt_bbox, txt_seg_points, bble_bbox, txt_class, alignment='', source_lang='')
            blk_list.append(text_region)
        if blk_list:
            blk_list = sort_regions(blk_list)

        mask = generate_mask(np.array(pil_img), blk_list)
        
        inpaint_input_img = self.inpainter(np.array(pil_img), mask, self.conf)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img) 
        inpaint_input_img = cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGB)
        return mask, inpaint_input_img
    
    def inpaint_api(self, image_path:str) -> dict:
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")

        mask, inpaint_input_img = self.inpaint(pil_image)
        image_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        inpaint_path = f"dataset/test_inpaint/{image_name}.jpg"
        Image.fromarray(inpaint_input_img).save(inpaint_path)

        return { "inpaint_path": inpaint_path }
    
    def inpaint_custom(self, pil_img, custom_mask):
        # print("Custom mask")
        inpaint_input_img = self.inpainter(np.array(pil_img), custom_mask, self.conf)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img) 
        inpaint_input_img = cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("inpainted.jpg", inpaint_input_img)
        return custom_mask, inpaint_input_img
