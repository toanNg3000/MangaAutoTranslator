from mltranslator import PROJECT_DIR
from mltranslator.modules.detection import TextDetector
from mltranslator.modules.jap_ocr import JapaneseReader
from mltranslator.modules.llm import GeminiLLM
from PIL import Image

my_text_detector = TextDetector()
my_jap_reader = JapaneseReader()
llm = GeminiLLM()

image_path = f"{PROJECT_DIR}/dataset/6.jpg"

if __name__ == "__main__":
    # text_detector = TextDetector()
    image = Image.open(image_path)
    yolo_img, list_detect_result = my_text_detector.get_detect_output(image)
    Image.fromarray(yolo_img).save('ocr_result.jpg')
    ocr_result = my_jap_reader.get_list_ocr(list_detect_result)
    print(ocr_result)

    joined_text_result = ""
    for i, t in enumerate(ocr_result):
        joined_text_result += f"{i}: {t}\n"
    response = llm.translate(joined_text_result)
    for chunk in response:
        print(chunk.text, end="")