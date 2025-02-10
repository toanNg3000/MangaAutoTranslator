import os
from pathlib import Path

_CURRENT_PROJECT_DIR = Path(os.path.dirname(__file__)).resolve()
PROJECT_DIR = Path(*_CURRENT_PROJECT_DIR.parts[:-1])