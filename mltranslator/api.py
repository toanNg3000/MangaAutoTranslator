from typing import List, Tuple, Any
import json
from pydantic import BaseModel, model_validator

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
