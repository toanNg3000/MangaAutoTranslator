from ultralytics import YOLO

class TextDetector(YOLO):
    def __init__(self):
        # model_path = 'models/text_detect/best.pt'
        model_path = 'C:/Users/User/Workspaces/GIT/MangaAutoTranslator/mltranslator/models/detection/text_detect/best.pt'
        super().__init__(model_path,)

image_path = "C:/Users/User/Workspaces/GIT/MangaAutoTranslator/dataset/part_1/00fd0625efee4c8186510fdf583fcb21.jpg"

if __name__ == "__main__":
    text_detector = TextDetector()
    results = text_detector.predict(image_path)
    print(results)
    for result in results:
        result.save_txt("output.txt")
        break
    pass