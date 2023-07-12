import threading

import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import io
import os
from uvicorn import run
import base64

import sys

from PIL import Image

from utils.pred_utils import predict_image, art_class_labels, device, diffusion_model_labels
from model.model import ArtVisionModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
# Load model
print("[INFO]: Loading model...")
art_vision_model = ArtVisionModel(len(art_class_labels)).to(device)
art_vision_model.load_state_dict(torch.load("./model/top_art_brain_model_state.pt", map_location=device))

# art_vision_model = torch.load("./model/art_brain_model_scripted.pt", map_location=device)
art_vision_model.eval()
print("[INFO]: Model loaded successfully.")

# Configurations
origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("art_home.html", {"request": request})


@app.post("/prediction/test")
async def art_brain_test(
        model_type: str = Form()):
    return {
        "mess": "passs" + model_type
    }


@app.post("/prediction/art")
async def art_brain_pred(
        model_type: str = Form(),
        img_file: UploadFile = File(...)):
    art_image = await img_file.read()

    art_image = Image.open(io.BytesIO(art_image))

    art_image.resize((256, 256))

    print("[INFO]: Prediction Started...")
    preds, attribution_scores, sorted_pred_index, pred_hm_image = predict_image(art_image, art_vision_model, hm_opacity=0.6)
    print("[INFO]: Prediction Completed...")

    with io.BytesIO() as art_img_byte_arr:
        # art_img_byte_arr = io.BytesIO()
        pred_hm_image.save(art_img_byte_arr, format='jpeg')

        pred_hm_image = str(base64.b64encode(art_img_byte_arr.getvalue()).decode("utf-8"))

    results_dict = {
        art_class_labels[pred_index]: preds[pred_index] for pred_index in sorted_pred_index
    }

    attribution_scores_dict = {
        diffusion_model_labels[attr_index]: attribution_scores[attr_index] for attr_index in range(len(attribution_scores))
    }

    attribution_scores_dict = {model_name: score for model_name, score in sorted(
        attribution_scores_dict.items(), key=lambda item: item[1], reverse=True
    )}

    return {
        "hm_img": pred_hm_image,
        "prediction_results": results_dict,
        "attribution_scores": attribution_scores_dict
    }


def run_server():
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # sys.setrecursionlimit(100000)
    threading.stack_size(10000000)
    thread = threading.Thread(target=run_server)
    thread.start()
