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
const heatmap_element = document.getElementById("heatmap");
const pred_display_elements = Array.from(document.getElementsByClassName("pred-results"));
const pred_display_fill_elements = Array.from(document.getElementsByClassName("pred-results-fill"));
const pred_display_label_elements = Array.from(document.getElementsByClassName("pred-results-label"));
const attr_display_elements = Array.from(document.getElementsByClassName("attr-results"));
const attr_display_fill_elements = Array.from(document.getElementsByClassName("attr-results-fill"));
const attr_display_label_elements = Array.from(document.getElementsByClassName("attr-results-label"));
const JET_RGB_COLORMAP_VALS = [
    0.2422, 0.1504, 0.6603, 0.25039, 0.165, 0.70761, 0.25777,
    0.18178, 0.75114, 0.26473, 0.19776, 0.79521, 0.27065, 0.21468,
    0.83637, 0.27511, 0.23424, 0.87099, 0.2783, 0.25587, 0.89907,
    0.28033, 0.27823, 0.9221, 0.28134, 0.3006, 0.94138, 0.28101,
    0.32276, 0.95789, 0.27947, 0.34467, 0.97168, 0.27597, 0.36668,
    0.9829, 0.26991, 0.3892, 0.9906, 0.26024, 0.41233, 0.99516,
    0.24403, 0.43583, 0.99883, 0.22064, 0.46026, 0.99729, 0.19633,
    0.48472, 0.98915, 0.1834, 0.50737, 0.9798, 0.17864, 0.52886,
    0.96816, 0.17644, 0.5499, 0.95202, 0.16874, 0.57026, 0.93587,
    0.154, 0.5902, 0.9218, 0.14603, 0.60912, 0.90786, 0.13802,
    0.62763, 0.89729, 0.12481, 0.64593, 0.88834, 0.11125, 0.6635,
    0.87631, 0.09521, 0.67983, 0.85978, 0.068871, 0.69477, 0.83936,
    0.029667, 0.70817, 0.81633, 0.0035714, 0.72027, 0.7917, 0.0066571,
    0.73121, 0.76601, 0.043329, 0.7411, 0.73941, 0.096395, 0.75,
    0.71204, 0.14077, 0.7584, 0.68416, 0.1717, 0.76696, 0.65544,
    0.19377, 0.77577, 0.6251, 0.21609, 0.7843, 0.5923, 0.24696,
    0.7918, 0.55674, 0.29061, 0.79729, 0.51883, 0.34064, 0.8008,
    0.47886, 0.3909, 0.80287, 0.43545, 0.44563, 0.80242, 0.39092,
    0.5044, 0.7993, 0.348, 0.56156, 0.79423, 0.30448, 0.6174,
    0.78762, 0.26124, 0.67199, 0.77927, 0.2227, 0.7242, 0.76984,
    0.19103, 0.77383, 0.7598, 0.16461, 0.82031, 0.74981, 0.15353,
    0.86343, 0.7406, 0.15963, 0.90354, 0.73303, 0.17741, 0.93926,
    0.72879, 0.20996, 0.97276, 0.72977, 0.23944, 0.99565, 0.74337,
    0.23715, 0.99699, 0.76586, 0.21994, 0.9952, 0.78925, 0.20276,
    0.9892, 0.81357, 0.18853, 0.97863, 0.83863, 0.17656, 0.96765,
    0.8639, 0.16429, 0.96101, 0.88902, 0.15368, 0.95967, 0.91346,
    0.14226, 0.9628, 0.93734, 0.12651, 0.96911, 0.96063, 0.10636,
    0.9769, 0.9839, 0.0805
];

const CLASSES_DISP_NAME = {
    'AI_LD_art_nouveau': "Art Nouveau - LD",
    'AI_LD_baroque': "Baroque - LD",
    'AI_LD_expressionism': "Expressionism - LD",
    'AI_LD_impressionism': "Impressionism - LD",
    'AI_LD_post_impressionism': "Post Impressionism - LD",
    'AI_LD_realism': "Realism - LD",
    'AI_LD_renaissance': "Renaissance - LD",
    'AI_LD_romanticism': "Romanticism - LD",
    'AI_LD_surrealism': "Surrealism - LD",
    'AI_LD_ukiyo-e': "Ukiyo-e  - LD",
    'AI_SD_art_nouveau': "Art Nouveau - SD",
    'AI_SD_baroque': "Baroque - SD",
    'AI_SD_expressionism': "Expressionism - SD",
    'AI_SD_impressionism': "Impressionism - SD",
    'AI_SD_post_impressionism': "Post Impressionism - SD",
    'AI_SD_realism': "Realism - SD",
    'AI_SD_renaissance': "Renaissance - SD",
    'AI_SD_romanticism': "Romanticism - SD",
    'AI_SD_surrealism': "Surrealism - SD",
    'AI_SD_ukiyo-e': "Ukiyo-e - SD",
    'art_nouveau': "Art Nouveau - Human",
    'baroque': "Baroque - Human",
    'expressionism': "Expressionism - Human",
    'impressionism': "Impressionism - Human",
    'post_impressionism': "Post Impressionism - Human",
    'realism': "Realism - Human",
    'renaissance': "Renaissance - Human",
    'romanticism': "Romanticism - Human",
    'surrealism': "Surrealism - Human",
    'ukiyo_e': "Ukiyo-e - Human"
}

const GEN_MODEL_NAME = {
    "latent_diffusion": "Latent Diffusion",
    "standard_diffusion": "Standard Diffusion",
    "human": "Human"
}

const CLASSES = Object.values(CLASSES_DISP_NAME)
const GEN_MODELS = Object.values(GEN_MODEL_NAME)

const IMG_SIZE = 224

//const clearButton = document.getElementById("clear-button");

image_upload.addEventListener("change", showImgPreview)
pred_submit_btn_element.addEventListener("click", sendImage)


// This function displays a preview of the uploaded image
function showImgPreview() {
    if (image_upload.files.length > 0) {
        var src = URL.createObjectURL(image_upload.files[0]);
        image_preview.src = src;
        image_preview.style.display = "inline-block";
        upload_btn_element.style.display = "none";
        pred_submit_btn_element.style.display = "block";
    }
}

// This function send the input image to the prediction function
function sendImage() {

    // Displaying loader
    pred_loading_element.style.display = "block"
    pred_submit_btn_element.style.display = "none"
    const predFormData = new FormData(pred_form_element);
    let img_data = predFormData.get("img_file")
    console.log(img_data.data)
    getPredictions()


}

// This function produces the predictions results and display them in the UI
async function getPredictions() {
    // Load art model
    const art_brain_model = await tf.loadLayersModel('model/modeljs/model.json');

    // Load image into a hidden canvas
    const uploadImage = new Image();
    uploadImage.src = image_preview.src
    image_preview_canvas.width = uploadImage.width
    image_preview_canvas.height = uploadImage.height
    const ctx = image_preview_canvas.getContext("2d");
    ctx.drawImage(uploadImage, 0, 0);


    // Image for model input
    let image = tf.image.resizeBilinear(tf.browser.fromPixels(
        image_preview_canvas
    ), [IMG_SIZE, IMG_SIZE]).toFloat();

    // Get original image in tensor form
    const org_image = image.reshape([1, 224, 224, 3]);

    const img_values = tf.moments(image, [0, 1, 2]);

    // Standardise the input image
    image = image.sub(img_values.mean).div(img_values.variance.sqrt()).reshape([1, 224, 224, 3]);

    // Get model prediction
    const prediction_result = await art_brain_model.predict(image);
    let attribution_scores = getAttributionScores(prediction_result.dataSync())

    // Filtering top 3 predictions
    const {values, indices} = await tf.topk(prediction_result, 3);

    let top_pred_values = values.dataSync()
    let top_pred_indices = indices.dataSync()
    let pred_results = prediction_result.dataSync() // TODO: Use for calculating the attribution scores

    let hm_pred_res = getGradCamPrediction(art_brain_model, top_pred_indices[0], image, org_image)
    let pred_heatmap = hm_pred_res[1].squeeze()
    let heatmap = hm_pred_res[0].squeeze()

    // Removing Loader
    pred_loading_element.style.display = "none"
    upload_btn_element.style.display = "block"

    // Draw heatmap image on a canvas
    const hm_canvas = document.createElement('canvas');
    hm_canvas.width = pred_heatmap.shape.width
    hm_canvas.height = pred_heatmap.shape.height
    await tf.browser.toPixels(pred_heatmap, hm_canvas);

    // Draw heatmap on a canvas
    const heatmap_canvas = document.createElement('canvas');
    heatmap_canvas.width = heatmap.shape.width
    heatmap_canvas.height = heatmap.shape.height
    await tf.browser.toPixels(heatmap, heatmap_canvas);

    pred_res_card_element.style.display = "flex"

    // Conversion to base64
    heatmap_image_element.src = hm_canvas.toDataURL("image/jpeg")
    heatmap_element.src = heatmap_canvas.toDataURL("image/jpeg")

    // Set the UI values
    top_pred_values.forEach(function (pred_value, index) {
        pred_display_label_elements.at(index).innerHTML = CLASSES[top_pred_indices[index]]

        let pred_result = Math.round(pred_value * 100)
        pred_display_elements.at(index).ariaValueNow = pred_result.toString()
        pred_display_fill_elements.at(index).innerHTML = pred_result + "%"
        pred_display_fill_elements.at(index).style.width = pred_result + "%"
    });

    attribution_scores[0].forEach(function (sorted_attr_index, index) {
        attr_display_label_elements.at(index).innerHTML = GEN_MODELS[sorted_attr_index]

        let pred_result = Math.round(attribution_scores[1][sorted_attr_index] * 100)
        attr_display_elements.at(index).ariaValueNow = pred_result.toString()
        attr_display_fill_elements.at(index).innerHTML = pred_result + "%"
        attr_display_fill_elements.at(index).style.width = pred_result + "%"
    });

}

// This function calculates the attribution scores
function getAttributionScores(preds) {
    let attr_scores = [
        preds.slice(0, 10).reduce((pred_sum, pred) => pred_sum + pred, 0),
        preds.slice(10, 20).reduce((pred_sum, pred) => pred_sum + pred, 0),
        preds.slice(20, 30).reduce((pred_sum, pred) => pred_sum + pred, 0)
    ]
    console.log(attr_scores)

    const attr_scores_sort_indices = Array.from(attr_scores.keys()).sort(
        (idx_0,idx_1) => attr_scores[idx_1]-attr_scores[idx_0]
    )
    return [attr_scores_sort_indices, attr_scores]
}

// This function generates the colourmap required for the Grad-CAM
function generateColMap(x) {

    return tf.tidy(() => {
        // Get normalized x.
        const EPSILON = 1e-5;
        const x_val_range = x.max().sub(x.min());
        const x_val_norm = x.sub(x.min()).div(x_val_range.add(EPSILON));
        const x_norm_data = x_val_norm.dataSync();

        const height = x.shape[1];
        const width = x.shape[2];
        const buffer = tf.buffer([1, height, width, 3]);

        const col_map_size = JET_RGB_COLORMAP_VALS.length / 3;

        for (let rows = 0; rows < height; ++rows) {
            for (let cols = 0; cols < width; ++cols) {
                const pixelValue = x_norm_data[rows * width + cols];
                const row = Math.floor(pixelValue * col_map_size);
                buffer.set(JET_RGB_COLORMAP_VALS[3 * row], 0, rows, cols, 0);
                buffer.set(JET_RGB_COLORMAP_VALS[3 * row + 1], 0, rows, cols, 1);
                buffer.set(JET_RGB_COLORMAP_VALS[3 * row + 2], 0, rows, cols, 2);
            }
        }
        return buffer.toTensor();
    });
}

// This functino implements the Grad-CAM
// GradCAM: https://doi.org/10.1109/ICCV.2017.74
// TF GradCAM: https://github.com/tensorflow/tfjs-examples/tree/master/visualize-convnet
function getGradCamPrediction(model, pred_class_index, input_image, org_image = 1) {

    // Last convolutional layer number
    let last_conv_layer_index = 151
    const last_conv_layer = model.layers[last_conv_layer_index];

    // Get last convolutional layer of the model
    const last_conv_layer_out = last_conv_layer.output;

    // Activation calculation model
    const act_model = tf.model({inputs: model.inputs, outputs: last_conv_layer_out});

    // New sub model that calculates gradients of the last convolutional layer w.r.t the final prediction
    const act_model_input = tf.input({shape: last_conv_layer_out.shape.slice(1)});
    last_conv_layer_index++;
    let y = act_model_input;
    while (last_conv_layer_index < model.layers.length) {
        y = model.layers[last_conv_layer_index++].apply(y);
    }
    const grad_model = tf.model({inputs: act_model_input, outputs: y});

    return tf.tidy(() => {
        // This functions calculates activations and gradients w.r.t a given class prediction
        const conv_wrt_class_output = (input) =>
            grad_model.apply(input, {training: true}).gather([pred_class_index], 1);

        // Gradient Calculation Model
        const grad_cal_function = tf.grad(conv_wrt_class_output);

        // Activation Calculation
        const last_conv_out_vals = act_model.apply(input_image);

        // Gradient Calculation
        const pred_grad_vals = grad_cal_function(last_conv_out_vals);

        // Pooling gradient values
        const pooled_gad_vals = tf.mean(pred_grad_vals, [0, 1, 2]);

        // Weighing the activations based on the gradients
        const weighted_conv_vals = last_conv_out_vals.mul(pooled_gad_vals);

        // Final pooling to get 2D map
        let grad_heatmap = weighted_conv_vals.mean(-1);

        // Post-processing of the map

        var img_values = tf.moments(grad_heatmap, [0, 1, 2])
        grad_heatmap = grad_heatmap.sub(img_values.mean).div(img_values.variance.sqrt());
        grad_heatmap = grad_heatmap.relu();
        grad_heatmap = grad_heatmap.div(grad_heatmap.max()).expandDims(-1);

        // Resizing the heatmap to match the input image
        grad_heatmap = tf.image.resizeBilinear(grad_heatmap, [org_image.shape[1], org_image.shape[2]]);

        // 'JET' colour map creation
        grad_heatmap = generateColMap(grad_heatmap);

        grad_heatmap = grad_heatmap.div(grad_heatmap.max()).relu()

        org_image = org_image.div(255);
        return [grad_heatmap, org_image];
    });
}

