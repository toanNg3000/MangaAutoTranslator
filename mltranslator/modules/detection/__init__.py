from ultralytics import YOLO

from mltranslator import PROJECT_DIR

from mltranslator.util.helper import *
import numpy as np
import cv2
import PIL

class TextDetector():
    def __init__(self, yolo_model_path:str = f'{PROJECT_DIR}/mltranslator/models/detection/best.pt', verbose:bool=False):
        # model_path = 'models/text_detect/best.pt'

        self.yolo_model_path = yolo_model_path
        self.verbose = verbose
        self.yolo_model = YOLO(yolo_model_path,verbose=verbose)

    def get_detect_output(self, image: PIL.ImageFile):
        list_sliced_images = split_image(image)
        list_sliced_images_size = []

        for i, sliced_image in enumerate(list_sliced_images):
            list_sliced_images_size.append(sliced_image.size)

        cumulative_heights = [cumulative_height(list_sliced_images_size[:i+1]) for i in range(len(list_sliced_images_size))]
        yolo_dict = []
        # start_time = time.time()
        for i, _ in enumerate(list_sliced_images):
            result = self.yolo_model.predict(list_sliced_images[i], device='cpu', half=False, conf=0.5, iou=0.6, augment=False, verbose=False)[0]
            for box in result.boxes:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
                if i > 0:
                    ymin += cumulative_heights[i-1]
                    ymax += cumulative_heights[i-1]
                yolo_dict.append({f"image_{i}":[xmin, ymin, xmax, ymax]})

        yolo_img = np.array(image)
        h, w, _ = yolo_img.shape
        inpainted_img = np.copy(yolo_img)
        # final_mask = np.zeros((h, w), dtype=np.uint8)  # Start with a black mask

        
        yolo_boxes = merge_bounding_boxes(yolo_dict)
        ocr_padding = 4
        ocr_padding_top_bottom = ocr_padding // 2
        # inpaint_padding = 30
        # inpaint_padding_top_bottom = 30

        # init payload
        list_result = []
        for idx, box in enumerate(yolo_boxes):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(yolo_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            # Add the index at the top of the bounding box
            cv2.putText(yolo_img, str(idx), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            x1_ocr = max(int(xmin) - ocr_padding, 0)  # Ensuring the value doesn't go below 0
            y1_ocr = max(int(ymin) - ocr_padding_top_bottom, 0)  # Adding padding to the top
            x2_ocr = min(int(xmax) + ocr_padding, w)  # Adjust according to the image width
            y2_ocr = min(int(ymax) + ocr_padding_top_bottom, h)  # Adding padding to the bottom

            # x1_inpaint = max(int(xmin) - inpaint_padding, 0)  # Ensuring the value doesn't go below 0
            # y1_inpaint = max(int(ymin) - inpaint_padding_top_bottom, 0)  # Adding padding to the top
            # x2_inpaint = min(int(xmax) + inpaint_padding, w)  # Adjust according to the image width
            # y2_inpaint = min(int(ymax) + inpaint_padding_top_bottom, h)  # Adding padding to the bottom
            
            ocr_cropped_image = inpainted_img[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
            list_result.append(ocr_cropped_image)
        cv2.putText(yolo_img, "YOLO", (w//2, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        return yolo_img, list_result
