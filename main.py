from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import io
import os
from uvicorn import run
import base64

from PIL import Image, ImageOps

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load model

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


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


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
    art_image = ImageOps.expand(art_image, border=10, fill=(70, 70, 70))

    art_img_byte_arr = io.BytesIO()
    art_image.save(art_img_byte_arr, format='jpeg')

    hm_image = str(base64.b64encode(art_img_byte_arr.getvalue()).decode("utf-8"))

    return {
        "hm_img": hm_image,
        "prediction_results": {
            "class1": 40,
            "class2": 30,
            "class3": 20,
            "class4": 10
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="0.0.0.0", port=port)
