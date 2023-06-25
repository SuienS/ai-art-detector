

const PRED_URL = "/prediction/art"
const image_upload = document.getElementById("image-file-input")
const pred_form_element = document.getElementById("pred-form-data");
const upload_btn_element = document.getElementById("upload-btn");
const pred_submit_btn_element = document.getElementById("pred-submit-btn");
const pred_loading_element = document.getElementById("pred-loading-button");
const image_preview = document.getElementById("img-upload-file")
const image_preview_canvas = document.getElementById("image-upload-canvas")
const pred_res_card_element = document.getElementById("pred-results-card");
const heatmap_image_element = document.getElementById("heatmap-image");
const pred_display_elements = Array.from(document.getElementsByClassName("pred-results"));
const pred_display_fill_elements = Array.from(document.getElementsByClassName("pred-results-fill"));
const pred_display_label_elements = Array.from(document.getElementsByClassName("pred-results-label"));

const IMG_SIZE = 224

//const clearButton = document.getElementById("clear-button");

image_upload.addEventListener("change", showImgPreview)
pred_submit_btn_element.addEventListener("click", sendImage)

const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./model/art_brain_onnx_model.onnx");

function showImgPreview() {
    if (image_upload.files.length > 0) {
        var src = URL.createObjectURL(image_upload.files[0]);
        image_preview.src = src;
        image_preview.style.display = "inline-block";
        upload_btn_element.style.display = "none";
        pred_submit_btn_element.style.display = "block";
    }
}

async function postImage(url, predFormData) {

    // Default options are marked with *
    return await fetch(url, {
        method: "POST", // *GET, POST, PUT, DELETE, etc.
        //mode: "cors", // no-cors, *cors, same-origin
        // headers: {
        //     "Content-Type": "multipart/form-data"
        // },
        body: predFormData, // body formData type must match "Content-Type" header
    }); // parses JSON response into native JavaScript objects
}

function sendImage() {

    // Displaying loader
    pred_loading_element.style.display = "block"
    pred_submit_btn_element.style.display = "none"
    const predFormData = new FormData(pred_form_element);
    let img_data = predFormData.get("img_file")
    console.log(img_data.data)
    getPredictions()


}

async function getPredictions() {
    // Get the predictions for the canvas data.

    var uploadImage = new Image();
    uploadImage.src = image_preview.src

    const ctx = image_preview_canvas.getContext("2d");

    ctx.drawImage(uploadImage, 0, 0);
    const img_data = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
    console.log(uploadImage)

    const input = new Tensor(new Float32Array(img_data.data), "float32");
    const input_feat = new Tensor(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]), "float32");

    const outputMap = await sess.run([input, input_feat]);
    const outputTensor = outputMap.values().next().value;
    const predictions = outputTensor.data;
    const maxPrediction = Math.max(...predictions);

    console.log(maxPrediction)

    // for (let i = 0; i < predictions.length; i++) {
    //   const element = document.getElementById(`prediction-${i}`);
    //   element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    //   element.className =
    //     predictions[i] === maxPrediction
    //       ? "prediction-col top-prediction"
    //       : "prediction-col";
    // }
}

//inmg src="data:image/jpeg;base64,{{ hm_img }}"
//https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
loadingModelPromise.then(() => {
    console.log("Ready")
})