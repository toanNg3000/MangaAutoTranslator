import os
import torch
import typing
from mltranslator import PROJECT_DIR

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

model_type = "vit_t"
sam_checkpoint = f"{PROJECT_DIR}/mltranslator/models/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

class MobileSam():
    def __init__(
        self,
        sam_checkpoint,
        model_type,
        device:typing.Literal["cuda", "cpu"]="cpu",
    ):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.model = self.model.to(device)
        self.model.eval()
        pass

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()

predictor = SamPredictor(mobile_sam)
predictor.set_image(<your_image>)
masks, _, _ = predictor.predict(<input_prompts>)