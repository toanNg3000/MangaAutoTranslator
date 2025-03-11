import io
import os
from typing import List, Tuple

import PIL
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from mltranslator import PROJECT_DIR
from mltranslator.modules.detection import TextDetector
from mltranslator.modules.inpainting.inpaintor import Inpaintor
from mltranslator.modules.jap_ocr import JapaneseReader
from mltranslator.modules.llm import GeminiLLM


class FullProcessRequest(BaseModel):
    image_path: str


class OCRRequest(BaseModel):
    image_path: str
    list_bboxes: List[
        Tuple[int, int, int, int]
    ]  # List of bounding boxes as lists of integers


class TranslateRequest(BaseModel):
    input_texts: List[str]


class InpaintRequest(BaseModel):
    image_path: str


# Create FastAPI app
app = FastAPI(title="Text Detection API")

# Enable CORS for all endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize text detector
text_detector = TextDetector()
japanese_reader = JapaneseReader()
llm = GeminiLLM()
inpanitor = Inpaintor()

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/detect")
async def detect_text(file: UploadFile = File(...)):
    """
    Endpoint for text detection in uploaded images

    - Accepts a single image file
    - Returns JSON with bounding box coordinates
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents))

    # Perform detection
    try:
        list_bboxes = text_detector.get_detect_output_api(pil_image)
        return {"data": list_bboxes}

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
        ocr_results = japanese_reader.get_list_orc_api(
            request.image_path, request.list_bboxes
        )
        return {"data": ocr_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
async def translate(request: TranslateRequest):
    """
    Endpoint for performing OCR on specified bounding boxes

    Expects:
    - list_texts: list of dectected text.

    Returns:
    - Translated text
    """
    try:
        prompt_input = ""
        for i, t in enumerate(request.input_texts):
            prompt_input += f"{i}: {t}\n"
        prompt_input += "\n"
        translate_results = llm.translate(prompt_input)

        translate_results = translate_results.text
        translate_results = translate_results.replace("<translate>", "").replace(
            "</translate>", ""
        )
        translate_results = translate_results.split("\n")

        results = {}
        for result in translate_results:
            result = result.strip()
            if not result:
                continue

            first_colons_idx = result.find(": ")
            idx = result[:first_colons_idx]
            text = result[first_colons_idx + 2 :]
            results[idx] = text

        return {"data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint")
async def perform_inpaint(request: InpaintRequest):
    """
    Endpoint for performing OCR on specified bounding boxes

    Expects:
    - image_path: Path to the image file

    Returns:
    - Output path to the inpainted image
    """
    try:
        inpaint_result = inpanitor.inpaint_api(
            request.image_path,
        )
        return {"data": inpaint_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/full_process")
async def perform_full_process(request: FullProcessRequest):
    """
    # desired json body
    {
        "data": {
            "<image-id>": {
                "data": {
                    "1": {
                        "bbox": Tuple[int, int, int, int],
                        "ocr_text": str,
                        "translate_text": str,
                    },
                    "2": {
                        "bbox": Tuple[int, int, int, int],
                        "ocr_text": str,
                        "translate_text": str,
                    }
                },
                "inpaint_path": str,
            }
        }
    }
    """

    # read image
    try:
        pil_image = PIL.Image.open(request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        list_bboxes = text_detector.get_detect_output_api(pil_image)

        # TODO: use cached image instead of loading again
        # ocr
        ocr_results = japanese_reader.get_list_orc_api(
            request.image_path,
            list_bboxes,
        )

        ocr_texts = [ocr_result["text"] for ocr_result in ocr_results.values()]

        # translate
        # TODO: refactor this
        prompt_input = ""
        for i, t in enumerate(ocr_texts):
            prompt_input += f"{i}: {t}\n"
        prompt_input += "\n"
        translate_texts = llm.translate(prompt_input)

        translate_texts = translate_texts.text
        translate_texts = translate_texts.replace("<translate>", "").replace(
            "</translate>", ""
        )
        translate_texts = translate_texts.split("\n")

        translate_results = {}
        for result in translate_texts:
            result = result.strip()
            if not result:
                continue

            first_colons_idx = result.find(": ")
            idx = result[:first_colons_idx]
            text = result[first_colons_idx + 2 :]
            translate_results[idx] = text

        # inpaint
        inpaint_path = inpanitor.inpaint_api(request.image_path)
        inpaint_path = inpaint_path["inpaint_path"]

        image_datas = {}
        for i, (bbox, ocr, translation) in enumerate(
            zip(list_bboxes, ocr_texts, translate_results.values())
        ):
            image_data = {
                "bbox": bbox,
                "ocr_text": ocr,
                "translation": translation,
            }
            image_datas[i] = image_data

        results = {"data": image_datas, "inpaint_path": inpaint_path}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/full_process_upload")
async def perform_full_process_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # read image
    try:
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        list_bboxes = text_detector.get_detect_output_api(pil_image)

        # TODO: use cached image instead of loading again
        # ocr
        ocr_results = japanese_reader.get_list_orc_from_img_api(
            pil_image,
            list_bboxes,
        )

        ocr_texts = [ocr_result["text"] for ocr_result in ocr_results.values()]

        # translate
        # TODO: refactor this
        prompt_input = ""
        for i, t in enumerate(ocr_texts):
            prompt_input += f"{i}: {t}\n"
        prompt_input += "\n"
        translate_texts = llm.translate(prompt_input)

        translate_texts = translate_texts.text
        translate_texts = translate_texts.replace("<translate>", "").replace(
            "</translate>", ""
        )
        translate_texts = translate_texts.split("\n")

        translate_results = {}
        for result in translate_texts:
            result = result.strip()
            if not result:
                continue

            first_colons_idx = result.find(": ")
            idx = result[:first_colons_idx]
            text = result[first_colons_idx + 2 :]
            translate_results[idx] = text

        # inpaint
        inpaint_path = inpanitor.inpaint_from_image_api(pil_image)
        inpaint_path = inpaint_path["inpaint_path"]

        image_datas = {}
        for i, (bbox, ocr, translation) in enumerate(
            zip(list_bboxes, ocr_texts, translate_results.values())
        ):
            image_data = {
                "bbox": bbox,
                "ocr_text": ocr,
                "translation": translation,
            }
            image_datas[i] = image_data

        results = {"data": image_datas, "inpaint_path": inpaint_path}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


# Optional: Add a health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Text Detection API is running"}


# To run: uvicorn fast_api:app --reload
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)