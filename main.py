import os
import numpy as np

from mltranslator import PROJECT_DIR
from mltranslator.modules.detection import TextDetector
from mltranslator.modules.jap_ocr import JapaneseReader
from mltranslator.modules.llm import GeminiLLM
from PIL import Image

my_text_detector = TextDetector()
my_jap_reader = JapaneseReader()
llm = GeminiLLM()


supported_ext = ['.jpg', '.png', '.jpeg']
input_dir = os.path.join(PROJECT_DIR, "dataset", "in")
output_dir = os.path.join(PROJECT_DIR, "dataset", "out")

if __name__ == "__main__":
    # text_detector = TextDetector()
    list_filepaths = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir)]

    # main flow
    ocr_results = {}
    for i, image_path in enumerate(list_filepaths):
        file_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(file_name)
        print(f"Processing {i+1}/{len(list_filepaths)}: {file_name}", end="")

        output_path = os.path.join(output_dir, f"{file_name}.jpg")
        if file_ext.lower() in supported_ext:
            image = Image.open(image_path)
            image = image.convert("RGB")
            yolo_img, list_detect_result = my_text_detector.get_detect_output(image)
            Image.fromarray(yolo_img).save(output_path)
            ocr_result = my_jap_reader.get_list_ocr(list_detect_result)
            ocr_results[image_path] = ocr_result
            # print(ocr_result)
        print("... Done!")

        # break

    prompt_input = ""
    for image_path, ocr_result in ocr_results.items():
        prompt_input += f"{image_path}:\n"
        for j, t in enumerate(ocr_result):
            prompt_input += f"{j}: {t}\n"
        prompt_input += "\n"
    
    print(prompt_input)
        
    # response = llm.translate(prompt_input)
    # for chunk in response:
    #     print(chunk.text, end="")

        # only run once, remove to run all
