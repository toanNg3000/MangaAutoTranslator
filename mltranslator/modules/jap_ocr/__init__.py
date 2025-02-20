from typing import List
from PIL import Image
from manga_ocr import MangaOcr
from mltranslator import PROJECT_DIR
from mltranslator.utils.helper import set_image_dpi

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