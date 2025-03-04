import os
import numpy as np
import cv2
import PIL
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from mltranslator.modules.detection import TextDetector
from mltranslator.modules.jap_ocr import JapaneseReader
from mltranslator.modules.llm import GeminiLLM
import io
from fastapi import Request
from pydantic import BaseModel
from typing import List, Tuple


# Assuming these are imported from your existing project
from mltranslator import PROJECT_DIR
from mltranslator.utils.helper import split_image, cumulative_height, merge_bounding_boxes

class OCRRequest(BaseModel):
    image_path: str
    list_bboxes: List[Tuple[int,int,int,int]]  # List of bounding boxes as lists of integers

class TranslateRequest(BaseModel):
    input_texts: List[str]

# Create FastAPI app
app = FastAPI(title="Text Detection API")

# Initialize text detector
text_detector = TextDetector()
japanese_reader = JapaneseReader()
llm = GeminiLLM()

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/detect")
async def detect_text(file: UploadFile = File(...)):
    """
    Endpoint for text detection in uploaded images
    
    - Accepts a single image file
    - Returns JSON with bounding box coordinates
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents))
    
    # Perform detection
    try:
        list_bboxes = text_detector.get_detect_output_api(pil_image)
        return list_bboxes
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/ocr")
async def perform_ocr(request: OCRRequest):
    """
    Endpoint for performing OCR on specified bounding boxes
    
    Expects:
    - image_path: Path to the image file
    - list_bboxes: List of bounding boxes [xmin, ymin, xmax, ymax]
    
    Returns:
    - OCR results with text for each bounding box
    """
    try:
        # Perform OCR
        ocr_results = japanese_reader.get_list_orc_api(
            # data["image_path"], 
            # data["list_bboxes"]
            request.image_path, 
            request.list_bboxes
        )
        return ocr_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate(request: TranslateRequest):
    """
    Endpoint for performing OCR on specified bounding boxes
    
    Expects:
    - list_texts: list of dectected text.
    
    Returns:
    - OCR results with text for each bounding box
    """
    try:
        prompt_input = ""
        for i, t in enumerate(request.input_texts):
            prompt_input += f"{i}: {t}\n"
        prompt_input += "\n"
        print(prompt_input)
        # Perform OCR
        translate_results = llm.translate(prompt_input)

        translate_results = translate_results.text
        translate_results = translate_results.replace("<translate>", "").replace("</translate>", "")
        translate_results = translate_results.split("\n")
        
        results = {}
        for result in translate_results:
            result = result.strip()
            if not result:
                continue

            first_colons_idx = result.find(": ")
            idx = result[:first_colons_idx]
            text = result[first_colons_idx+2:]
            results[idx] = text

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Add a health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Text Detection API is running"}

# To run: uvicorn main:app --reload


'''
    "Ｃａｒｒｅｓ ",
    "Ｒｏｎｓ ",
    "しっかし驚いたな～ ",
    "それで俺の出番ってわけか ",
    "晶がまさか文化祭で主役をやるってな ",
    "はい ",
    "なにかアドバイスをもらえないかと思いまして．．． ",
    "運さんは晶の実父であり俳優でもある ",
    "ドラマに何度か出演したことがあるらしい ",
    "きっと晶の力になってくれるはずだ ",
    "ゴホンッ ",
    "いいか晶役者の真髄っていうのはなー ",
    "ここ最近も役者の仕事で忙しいと聞いていた ",
    "あっ "
'''

'''
{
  "1": "Rons",
  "2": "Thật là ngạc nhiên!",
  "3": "Vậy thì đến lượt tôi rồi sao",
  "4": "Không ngờ Akari lại đóng vai chính trong lễ hội văn hóa",
  "5": "Vâng",
  "6": "Tôi muốn hỏi xem có lời khuyên nào không...",
  "7": "Ông Un là cha ruột của Akari và cũng là một diễn viên",
  "8": "Có vẻ như ông ấy đã tham gia diễn xuất trong một vài bộ phim truyền hình",
  "9": "Chắc chắn ông ấy sẽ giúp Akari",
  "10": "(Rõ họng)",
  "11": "Này Akari, tinh hoa của một diễn viên đó là...",
  "12": "Gần đây tôi nghe nói ông ấy bận rộn với công việc diễn xuất",
  "13": "À",
  " 0": "Carres",
  "": ""
}
'''