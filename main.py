import threading

import torch
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import io
import os
import pickle

from uvicorn import run
import base64

from PIL import Image

from constants import ART_CLASS_LABELS, DIFFUSION_MODEL_LABELS
from utils.inferencing import InferencingService
from model.model import AttentionConvNeXt

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")
# Load model
print("[INFO]: Loading model...")


# preprocess_transforms = imagenet_weights.transforms()
# with open('./model/preprocess_transforms.pt', 'wb') as f:
#     pickle.dump(preprocess_transforms, f)

with open('./model/preprocess_transforms.pt', 'rb') as f:
    preprocess_transforms = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

art_brain_model = AttentionConvNeXt(len(ART_CLASS_LABELS)).to(device)
# art_vision_model.load_state_dict(torch.load("./model/top_art_brain_model.pt", map_location=device)['model_state_dict'])
# torch.save(art_vision_model.state_dict(), "./model/top_art_brain_model_state.pt")
art_brain_model.load_state_dict(torch.load("./model/artbrain_top_model_weights.pt", map_location=device))
# art_vision_model = torch.load("./model/art_brain_model_scripted.pt", map_location=device)
art_brain_model.eval()

inferencing_service = InferencingService(
    model=art_brain_model,
    preprocess_transforms=preprocess_transforms,
    device=device
)

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
        test_string: str = Form()):
    return {
        "message": "Pass: " + test_string
    }


@app.post("/prediction/art")
async def art_brain_pred(
        heatmap_type: str = Form(),
        img_file: UploadFile = File(...)):
    try:
        art_image = await img_file.read()

        art_image = Image.open(io.BytesIO(art_image))

        print("[INFO]: Prediction Started...")
        print("[INFO]: Heatmap type: ", heatmap_type)

        preds, attribution_scores, sorted_pred_index, pred_hm_image, heatmap = inferencing_service.predict_image(
            art_image, heatmap_type
        )
        print("[INFO]: Prediction Completed...")

        with io.BytesIO() as art_img_byte_arr:
            pred_hm_image.save(art_img_byte_arr, format='jpeg')

            pred_hm_image = str(base64.b64encode(art_img_byte_arr.getvalue()).decode("utf-8"))

        with io.BytesIO() as hm_byte_arr:
            heatmap.save(hm_byte_arr, format='jpeg')

            heatmap = str(base64.b64encode(hm_byte_arr.getvalue()).decode("utf-8"))

        results_dict = {
            ART_CLASS_LABELS[pred_index]: preds[pred_index] for pred_index in sorted_pred_index
        }

        attribution_scores_dict = {
            DIFFUSION_MODEL_LABELS[attr_index]: attribution_scores[attr_index] for attr_index in
            range(len(attribution_scores))
        }

        attribution_scores_dict = {model_name: score for model_name, score in sorted(
            attribution_scores_dict.items(), key=lambda item: item[1], reverse=True
        )}

        return {
            "hm_img": pred_hm_image,
            "heatmap": heatmap,
            "prediction_results": results_dict,
            "attribution_scores": attribution_scores_dict
        }
    except Exception as e:
        # raise e
        return {"error": str(e)}


def run_server():
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # Increasing stack size for the application to run
    threading.stack_size(10000000)
    thread = threading.Thread(target=run_server)
    thread.start()
