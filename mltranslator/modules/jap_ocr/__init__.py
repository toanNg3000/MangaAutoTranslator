from typing import List, Tuple
from PIL import Image
from manga_ocr import MangaOcr
from mltranslator import PROJECT_DIR
from mltranslator.utils.helper import set_image_dpi
import numpy as np

class JapaneseReader:
    def __init__(self) -> None:
        self.model = MangaOcr(
            force_cpu=False,
            pretrained_model_name_or_path=f'{PROJECT_DIR}/mltranslator/models/jap_ocr'
        )

    def predict(self, np_image):
        try:
            pil_image = Image.fromarray(np_image)
            result = self.model(pil_image)
            if result is not None:
                return result
            return ""
        except:
            return ""
        
    def get_list_ocr(self, list_cropped_images)->List[str]:
        list_ocr_texts = []
        for cropped_image in list_cropped_images:
            txt = ""
            result = self.predict(set_image_dpi(cropped_image))
            txt += result + " "
            list_ocr_texts.append(txt)
        return list_ocr_texts

    def get_list_orc_api(self, image_path: str, list_bboxes: Tuple[int,int,int,int]):
        img = Image.open(image_path)
        img = img.convert("RGB")
        w, h = img.size
        np_img = np.array(img)
        ocr_padding = 4
        ocr_padding_top_bottom = ocr_padding // 2
        list_result = []

        for idx, box in enumerate(list_bboxes):
            xmin, ymin, xmax, ymax = box

            x1_ocr = max(int(xmin) - ocr_padding, 0)  # Ensuring the value doesn't go below 0
            y1_ocr = max(int(ymin) - ocr_padding_top_bottom, 0)  # Adding padding to the top
            x2_ocr = min(int(xmax) + ocr_padding, w)  # Adjust according to the image width
            y2_ocr = min(int(ymax) + ocr_padding_top_bottom, h)  # Adding padding to the bottom

            ocr_cropped_image = np_img[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
            list_result.append(ocr_cropped_image)

        list_ocr_text = self.get_list_ocr(list_result)

        results = {}
        for i, bboxes in enumerate(list_bboxes):
            results[i] = {
                "bboxes": bboxes,
                "text": list_ocr_text[i] # <--------
            }
        return { "ocr_results": results }