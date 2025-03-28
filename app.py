import io
import json
import os
from typing import Annotated, Any, List, Tuple

import numpy as np
import PIL
import torch
import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from starlette.responses import FileResponse

from mltranslator import PROJECT_DIR
from mltranslator.modules.detection import TextDetector
from mltranslator.modules.inpainting.inpaintor import Inpaintor
from mltranslator.modules.jap_ocr import JapaneseReader
from mltranslator.modules.llm import GeminiLLM
from mltranslator.utils.inpainting import create_mask_from_polygon


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


class InpaintRequestPolygon(BaseModel):
    image_path: str
    list_points: List[Tuple[int, int]]


class InpaintRequestPolygonUpload(BaseModel):
    list_points: List[Tuple[int, int]]

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


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
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
text_detector = TextDetector(device=DEVICE)
japanese_reader = JapaneseReader(device=DEVICE)
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
        ocr_results = japanese_reader.get_list_orc_from_path_api(
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
        results = llm.translate_api(request.input_texts)
        return {"data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint")
async def perform_inpaint(request: InpaintRequest):
    """
    Expects:
    - image_path: Path to the image file

    Returns:
    - Output path to the inpainted image
    """
    # read image
    try:
        pil_image = PIL.Image.open(request.image_path)
        pil_image = pil_image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        inpaint_path = inpanitor.inpaint_api(pil_image)
        return {"data": inpaint_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint-polygon")
async def perform_inpaint_polygon(request: InpaintRequestPolygon):
    """
    Expects:
    - image_path: Path to the image file
    - list_points: list of points in 2D spaces making up the polygon. E.x: [(0,0), (1,0), (1,1)]

    Returns:
    - Output path to the inpainted image
    """
    try:
        pil_image = PIL.Image.open(request.image_path)
        pil_image = pil_image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        mask = create_mask_from_polygon(pil_image.size, request.list_points)
        mask = np.array(mask)
        inpaint_path = inpanitor.inpaint_custom_mask_api(pil_image, mask)
        return {"data": inpaint_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inpaint-polygon-upload")
async def perform_inpaint_polygon_upload(
    file: Annotated[UploadFile, File()],
    request: Annotated[InpaintRequestPolygonUpload, Body()],
):
    """
    Expects:
    - file: uploaded file
    - list_points: list of points in 2D spaces making up the polygon. E.x: [(0,0), (1,0), (1,1)]

    Returns:
    - Output path to the inpainted image
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        pil_image = PIL.Image.open(io.BytesIO(contents))
        pil_image = pil_image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        mask = create_mask_from_polygon(pil_image.size, request.list_points)
        mask = np.array(mask)
        inpaint_path = inpanitor.inpaint_custom_mask_api(pil_image, mask)
        return {"data": inpaint_path}

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
        ocr_texts = japanese_reader.get_list_orc_from_path_api(
            request.image_path,
            list_bboxes,
        )

        # ocr_texts = [ocr_result["text"] for ocr_result in ocr_results.values()]

        # translate
        translated_texts = llm.translate_api(ocr_texts)

        # inpaint
        inpaint_path = inpanitor.inpaint_api(pil_image)

        image_datas = {}
        for i, (bbox, ocr, translation) in enumerate(
            zip(list_bboxes, ocr_texts, translated_texts)
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

        list_ocr_text = japanese_reader.get_list_orc_api(
            pil_image,
            list_bboxes,
        )

        translated_texts = llm.translate_api(list_ocr_text)

        inpaint_path = inpanitor.inpaint_api(pil_image)

        image_datas = {}
        for i, (bbox, ocr, translation) in enumerate(
            zip(list_bboxes, list_ocr_text, translated_texts)
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


@app.get("/images/{file_path:path}")
async def get_image(file_path: str):
    print("???", file_path)
    return FileResponse(f"{file_path}")


# Optional: Add a health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Text Detection API is running"}


# To run: uvicorn fast_api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
