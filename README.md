# MangaAutoTranslator

## Prerequisites

Python 3.11

## Instructions

**NOTE**: To use pytorch with GPU, you must install `pytorch` manually through it [official website](https://pytorch.org/get-started/locally/). If you decide to use GPU, this must be done first, otherwise just install the requirements below.

```sh
pip install -r requirements.txt
```

To run the program:

```sh
python main.py
```

Note: The translation module uses Gemini model. Simply set `GEMINI_API_KEY` environment variable to your Gemini api key before running the program.

```sh
export GEMINI_API_KEY="YOUR_API_KEY" # on linux
# or
set GEMINI_API_KEY=YOUR_API_KEY # on Windows
```

### Fast-API
simply run:

```sh
uvicorn fast_api:app
```

then go to the `/docs` pages