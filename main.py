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

from PIL import Image

from utils.pred_utils import predict_image, scaler, art_class_labels, device
from model.model import ArtVisionModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load model
print("[INFO]: Loading model...")
art_vision_model = ArtVisionModel(len(art_class_labels)).to(device)
art_vision_model.load_state_dict(torch.load("./model/art_vision_model_state.pt", map_location=device))
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
        "mess": "passs"+model_type
    }


@app.post("/prediction/art")
async def art_brain_pred(
        model_type: str = Form(),
        img_file: UploadFile = File(...)):
    art_image = await img_file.read()

    art_image = Image.open(io.BytesIO(art_image))

    art_image.resize((256, 256))

    preds, pred_index, pred_hm_image = predict_image(art_image, art_vision_model, scaler, hm_opacity=0.4)
    preds = preds.tolist()

    # art_image = image.img_to_array(art_image)
    #
    # pred_style_index, fusioned_hm_image = art_utils.make_prediction(
    #     art_image=art_image,
    #     art_style_model=art_style_model,
    #     cnn_end_conv_name=cnn_end_conv_name,
    #     hm_opacity=hm_opacity
    # )
    #
    # fusioned_hm_image = image.array_to_img(fusioned_hm_image)
    #

    art_img_byte_arr = io.BytesIO()
    pred_hm_image.save(art_img_byte_arr, format='jpeg')

    pred_hm_image = str(base64.b64encode(art_img_byte_arr.getvalue()).decode("utf-8"))

    sorted_pred_results = list(sorted(
        zip(preds[0], art_class_labels),
        reverse=True
    ))

    print(sorted_pred_results)

    results_dict = {}
    for pres_val, art_class in sorted_pred_results:
        results_dict[art_class] = pres_val

    return {
        "hm_img": pred_hm_image,
        "prediction_results": results_dict
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)
