import datetime
import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from mltranslator import PROJECT_DIR
from mltranslator.modules.detection import TextDetector
from mltranslator.modules.inpainting.lama import LaMa
from mltranslator.modules.inpainting.schema import Config
from mltranslator.utils.detection import combine_results
from mltranslator.utils.textblock import TextBlock, sort_regions

# TODO: this path should be configurable
INPAINT_DIR = os.path.join(PROJECT_DIR, "dataset/test_inpaint")


def generate_mask(img: np.ndarray, blk_list: List[TextBlock], default_kernel_size=5):
    h, w, *_ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # Start with a black mask

    for blk in blk_list:
        seg = blk.segm_pts
        # if blk.source_lang == 'en':
        #     default_kernel_size = 1
        kernel_size = default_kernel_size  # Default kernel size
        if blk.text_class == "text_bubble":
            # Access the bounding box coordinates
            bbox = blk.bubble_xyxy
            # Calculate the minimal distance from the mask to the bounding box edges
            min_distance_to_bbox = min(
                np.min(seg[:, 0]) - bbox[0],  # left side
                bbox[2] - np.max(seg[:, 0]),  # right side
                np.min(seg[:, 1]) - bbox[1],  # top side
                bbox[3] - np.max(seg[:, 1]),  # bottom side
            )
            # adjust kernel size if necessary
            if default_kernel_size >= min_distance_to_bbox:
                kernel_size = max(
                    5, int(min_distance_to_bbox - (0.2 * min_distance_to_bbox))
                )

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
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        # fmt: off
        self.text_detection = YOLO(f"{PROJECT_DIR}/mltranslator/models/detection/best.pt").to(self.device)
        self.text_segmentation = YOLO(f"{PROJECT_DIR}/mltranslator/models/inpainting/comic-text-segmenter.pt").to(self.device)

        self.text_detectionv2 = TextDetector(verbose=False)
        self.segmentation_model = YOLO(f"{PROJECT_DIR}/mltranslator/models/text_segment/best.pt", verbose=False).to(self.device)
        # fmt: on

        img_size_process = 512

        self.inpainter = LaMa(self.device)
        self.conf = Config(
            hd_strategy="Crop",
            hd_strategy_crop_trigger_size=img_size_process,
            hd_strategy_resize_limit=img_size_process,
            hd_strategy_crop_margin=2,
        )

    # deprecated
    def inpaint(self, pil_image: Image.Image):
        yolo_device = self.device
        text_detec_result = self.text_detection(
            pil_image, device=yolo_device, half=True, imgsz=640, conf=0.5, verbose=False
        )[0]
        txt_seg_result = self.text_segmentation(
            pil_image,
            device=yolo_device,
            half=True,
            imgsz=1024,
            conf=0.1,
            verbose=False,
        )[0]

        combined = combine_results(text_detec_result, txt_seg_result)

        blk_list: List[TextBlock] = []
        for txt_bbox, bble_bbox, txt_seg_points, txt_class in combined:
            text_region = TextBlock(
                txt_bbox,
                txt_seg_points,
                bble_bbox,
                txt_class,
                alignment="",
                source_lang="",
            )
            blk_list.append(text_region)
        if blk_list:
            blk_list = sort_regions(blk_list)

        mask = generate_mask(np.array(pil_image), blk_list)

        inpaint_input_img = self.inpainter(np.array(pil_image), mask, self.conf)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        inpaint_input_img = cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGB)
        return mask, inpaint_input_img

    def inpaint_v2(self, pil_image: Image.Image):
        bboxes = self.text_detectionv2.get_detect_output_api(pil_image)
        ocr_padding = 4
        ocr_padding_top_bottom = ocr_padding // 2
        np_original_img = np.array(pil_image)
        h, w = pil_image.size
        final_mask = np.zeros((w, h), dtype=np.uint8)

        for box in bboxes:
            xmin, ymin, xmax, ymax = box
            # fmt: off
            x1_ocr = max(int(xmin) - ocr_padding, 0)  # Ensuring the value doesn't go below 0
            y1_ocr = max(int(ymin) - ocr_padding_top_bottom, 0)  # Adding padding to the top
            x2_ocr = min(int(xmax) + ocr_padding, w)  # Adjust according to the image width
            y2_ocr = min(int(ymax) + ocr_padding_top_bottom, h)  # Adding padding to the bottom
            if x1_ocr >= x2_ocr:
                x1_ocr = xmin
                x2_ocr = xmax
            if y1_ocr >= y2_ocr:
                y1_ocr = ymin
                y2_ocr = ymax
            # fmt: on
            crop_img = np_original_img[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
            h_crop, w_crop, *_ = crop_img.shape
            seg_results = self.segmentation_model(
                crop_img,
                verbose=False,
            )
            for seg_result in seg_results:
                # Get array results
                masks = seg_result.masks.data
                boxes = seg_result.boxes.data
                # Extract classes
                clss = boxes[:, 5]
                # Get indices of results where class is 0 (people in COCO)
                indices = torch.where((clss == 0) | (clss == 1))
                # Use these indices to extract the relevant masks
                masks = masks[indices]
                # Scale for visualizing results
                mask = torch.any(masks, dim=0).int() * 255
                mask_np = mask.cpu().numpy().astype(np.uint8)
                pil_mask = Image.fromarray(mask_np).resize((w_crop, h_crop))
                final_mask[y1_ocr:y2_ocr, x1_ocr:x2_ocr] = np.array(pil_mask)

        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        final_mask = cv2.erode(final_mask, kernel)
        final_mask = cv2.dilate(final_mask, kernel, iterations=3)
        inpaint_input_img = self.inpainter(np.array(pil_image), final_mask, self.conf)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        inpaint_input_img = cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGB)
        return final_mask, inpaint_input_img

    def inpaint_from_path_api(self, image_path: str) -> dict:
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        inpaint_path = self.inpaint_api(pil_image)
        return inpaint_path

    def inpaint_api(self, pil_image: Image.Image) -> dict:
        pil_image_RGB = pil_image.convert("RGB")
        _, inpaint_input_img = self.inpaint_v2(pil_image_RGB)

        image_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(INPAINT_DIR, exist_ok=True)
        inpaint_path = os.path.join(INPAINT_DIR, f"{image_name}.jpg")

        Image.fromarray(inpaint_input_img).save(inpaint_path)

        return inpaint_path

    def inpaint_custom(self, pil_image: Image.Image, custom_mask: np.ndarray):
        inpaint_input_img = self.inpainter(np.array(pil_image), custom_mask, self.conf)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        inpaint_input_img = cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGB)

        return custom_mask, inpaint_input_img

    def inpaint_custom_mask_api(self, pil_image: Image.Image, custom_mask: np.ndarray):
        _, inpaint_input_img = self.inpaint_custom(pil_image, custom_mask)

        image_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(INPAINT_DIR, exist_ok=True)
        inpaint_path = os.path.join(INPAINT_DIR, f"{image_name}.jpg")
        Image.fromarray(inpaint_input_img).save(inpaint_path)

        return inpaint_path
