# ArtBrain: AI Art Detector

[![Paper](https://img.shields.io/badge/IEEE%20TMM-10.1109%2FTMM.2026.3743638-00629B)](https://doi.org/10.1109/TMM.2026.3743638)
[![arXiv](https://img.shields.io/badge/arXiv-2412.01512-b31b1b)](https://arxiv.org/abs/2412.01512)
[![Dataset](https://img.shields.io/badge/Kaggle-AI--ArtBench-20BEFF)](https://www.kaggle.com/datasets/ravidussilva/real-ai-art)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This is the official code for the ArtBrain application from the paper:

> **ArtBrain: An Explainable End-to-End Toolkit for Classification and Attribution of AI-Generated Art and Style**<br>
> Ravidu Suien Rammuni Silva, Ahmad Lotfi, Isibor Kennedy Ihianle, Golnaz Shahtahmassebi, Jordan J. Bird<br>
> *IEEE Transactions on Multimedia*, 2026. [[Paper]](https://doi.org/10.1109/TMM.2026.3743638) [[arXiv]](https://arxiv.org/abs/2412.01512)

ArtBrain takes an image of an artwork and tells you:

- **Who made it:** a human artist, Stable Diffusion or Latent Diffusion.
- **Which art style it is:** one of 10 styles, including Baroque, Impressionism, Surrealism and Ukiyo-e.
- **Why the model decided that:** a heatmap from [FM-G-CAM](https://arxiv.org/abs/2312.05975) or Grad-CAM that highlights the parts of the image the model relied on.

The classifier is **AttentionConvNeXt**, a ConvNeXt-based CNN that combines feature maps from several depths using a squeeze-and-excitation attention module. It was trained on [AI-ArtBench](https://www.kaggle.com/datasets/ravidussilva/real-ai-art), a dataset of 185,015 images: 125,015 AI-generated and 60,000 made by humans. AttentionConvNeXt reaches a macro F1-score of 0.869 across the 30 style-and-source classes and 0.999 for attributing an image to its source model.

![ArtBrain Web Application Screenshots](misc/ArtBrainApp.png?raw=true "ArtBrain Web Application")

## Contents

| Path | Description |
|---|---|
| [`AppFastAPI/`](AppFastAPI) | Server-side app. It runs the PyTorch AttentionConvNeXt model behind a FastAPI web interface and REST API, and generates FM-G-CAM / Grad-CAM heatmaps. |
| [`AppTFJS/`](AppTFJS) | On-device app. It runs a lightweight MobileNetV2 model in the browser with TensorFlow.js, so images never leave the device. |
| [`AppTFJS/standaloneHTML/`](AppTFJS/standaloneHTML) | A single HTML file containing the on-device app and all of its dependencies, for offline use. |
| [`ModelTrainingNotebooks/`](ModelTrainingNotebooks) | Model training notebooks (to be added). |
| [`Dockerfile`](Dockerfile) | Container image that serves both apps. |

## Installation

### 1. Prerequisites

- Python 3.10+ and `pip`
- Optional: an NVIDIA GPU with CUDA. The FastAPI app uses the GPU when one is available and falls back to the CPU otherwise. A GPU with at least 10 GB of VRAM is recommended.

### 2. Install dependencies

We recommend using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add the model files

The trained model weights are not included in this repository. Contact the [author](#author) to get them, then place them as follows:

```
AppFastAPI/model/
├── artbrain_top_model_weights.pt   # AttentionConvNeXt weights
└── preprocess_transforms.pt        # Image preprocessing transforms
AppTFJS/model/
└── modeljs/
    ├── model.json                  # TensorFlow.js MobileNetV2 model
    └── *.bin                       # Weight shards
```

## Usage

### Server-side app (FastAPI)

```bash
cd AppFastAPI
python3 main.py
```

Then open <http://localhost:8000>. The on-device app is also served at <http://localhost:8000/on-device-app>. To use a different port, set the `PORT` environment variable.

You can also call the prediction endpoint directly:

```bash
curl -X POST http://localhost:8000/prediction/art \
  -F "heatmap_type=FM-G-CAM" \
  -F "img_file=@/path/to/artwork.jpg"
```

`heatmap_type` is either `FM-G-CAM` or `G-CAM`. The response is JSON containing:

- `prediction_results`: scores for each style-and-source class, sorted from highest to lowest
- `attribution_scores`: scores for each source (Human, Stable Diffusion, Latent Diffusion)
- `hm_img` and `heatmap`: the heatmap overlaid on the image, and the heatmap on its own, both as base64-encoded JPEGs

### Docker

Add the model files first (step 3), then run:

```bash
docker build -t artbrain .
docker run -p 8000:8000 artbrain
# For GPU support (requires the NVIDIA Container Toolkit):
# docker run --gpus all -p 8000:8000 artbrain
```

### On-device app (TensorFlow.js)

Open [`AppTFJS/js_art_home.html`](AppTFJS/js_art_home.html) in a web browser, keeping the `model/` folder in the same place relative to the HTML file as shown above. Some browsers block local file access from `file://` pages. If the model doesn't load, serve the folder locally:

```bash
cd AppTFJS
python3 -m http.server 8080
# then open http://localhost:8080/js_art_home.html
```

## Dataset

AI-ArtBench is available on Kaggle: <https://www.kaggle.com/datasets/ravidussilva/real-ai-art>. It covers 10 art styles from three sources: human-made images from [ArtBench-10](https://github.com/liaopeiyuan/artbench), and images generated with Latent Diffusion and Stable Diffusion 2.1.

## Citation

If you use this code, the models or the AI-ArtBench dataset, please cite the paper:

```bibtex
@article{silva2026artbrain,
  author  = {Rammuni Silva, Ravidu Suien and Lotfi, Ahmad and Ihianle, Isibor Kennedy and Shahtahmassebi, Golnaz and Bird, Jordan J.},
  title   = {{ArtBrain}: An Explainable End-to-End Toolkit for Classification and Attribution of {AI}-Generated Art and Style},
  journal = {IEEE Transactions on Multimedia},
  year    = {2026},
  pages   = {1--11},
  doi     = {10.1109/TMM.2026.3743638}
}
```

The preprint is also available on arXiv:

```bibtex
@misc{silva2024artbrain,
  author        = {Rammuni Silva, Ravidu Suien and Lotfi, Ahmad and Ihianle, Isibor Kennedy and Shahtahmassebi, Golnaz and Bird, Jordan J.},
  title         = {{ArtBrain}: An Explainable End-to-End Toolkit for Classification and Attribution of {AI}-Generated Art and Style},
  year          = {2024},
  eprint        = {2412.01512},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2412.01512}
}
```

## Acknowledgements

This work began as an MSc Data Science research project at Nottingham Trent University.

## License

This project is released under the [MIT License](LICENSE).

## Author

- [Ravidu Suien Rammuni Silva](mailto:ravidus.ac@gmail.com)
