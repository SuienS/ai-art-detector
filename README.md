# ArtBrain: AI Art Detector
This repository contains code for the prototype application developed as a part of my research project in MSc Data Science at Nottingham Trent University. The repo contains two main applications; an cloud based application and on-device application. The preprint of this work available [here](https://doi.org/10.48550/arXiv.2412.01512).

![ArtBrain Web Application Screenshots](/misc/ArtBrainApp.png?raw=true "ArtBrain Web Application")

## Contents
 - AppFastAPI: ArtBrain developed using FastAPI.
 - AppTFJS: ArtBrain developed using TensorFlowJS.

## Installation

### 1. Pre-requisites

- Python 3.10+
- PIP
- CUDA supported GPU with at least 10GB VRAM
  - CUDA installation may also be required.

### 2. Install Dependencies
Run the following command to install the dependancies. Recommended to use
and virtual environment.

   ```
   pip install -r requirements.txt
   ```

### 3. Model file placement

Place the model files in their respective folders. Contact the author for the trained models.

### 4. Running the Application
To run the FastAPI application, run the following commands in the terminal.

```
cd AppFastAPI \
python3 main.py
```

After the execution of this line you can visit your localhost to use the application. Alternatively, you can use Docker to host the application using the Dockerfile provided.

To use the TensorFlowJS application, just open the [js_art_home.html](https://github.com/SuienS/ai-art-detector/blob/master/AppTFJS/js_art_home.html) file in a web browser. Make sure to have the model folders in the same directory as in the repository.

# Author
- [Ravidu Suien Rammuni Silva](mailto:ravidus.acv@gmail.com)
