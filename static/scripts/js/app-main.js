const PRED_URL = "/prediction/art"
const image_upload = document.getElementById("image-file-input")
const pred_form_element = document.getElementById("pred-form-data");
const upload_btn_element = document.getElementById("upload-btn");
const image_preview = document.getElementById("img-upload-file")
const pred_submit_btn_element = document.getElementById("pred-submit-btn");
const pred_loading_element = document.getElementById("pred-loading-button");
const pred_res_card_element = document.getElementById("pred-results-card");
const heatmap_image_element = document.getElementById("heatmap-image");
const pred_display_elements = Array.from(document.getElementsByClassName("pred-results"));
const pred_display_fill_elements = Array.from(document.getElementsByClassName("pred-results-fill"));
const pred_display_label_elements = Array.from(document.getElementsByClassName("pred-results-label"));
const attr_display_elements = Array.from(document.getElementsByClassName("attr-results"));
const attr_display_fill_elements = Array.from(document.getElementsByClassName("attr-results-fill"));
const attr_display_label_elements = Array.from(document.getElementsByClassName("attr-results-label"));

image_upload.addEventListener("change", showImgPreview)
pred_submit_btn_element.addEventListener("click", sendImage)


const CLASSES_DISP_NAME = {
    'AI_LD_art_nouveau':"Art Nouveau - LD",
    'AI_LD_baroque':"Baroque - LD",
    'AI_LD_expressionism':"Expressionism - LD",
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
    'art_nouveau': "Art Nouveau",
    'baroque': "Baroque",
    'expressionism':"Expressionism",
    'impressionism': "Impressionism",
    'post_impressionism': "Post Impressionism",
    'realism': "Realism",
    'renaissance': "Renaissance",
    'romanticism': "Romanticism",
    'surrealism': "Surrealism",
    'ukiyo_e': "Ukiyo-e"
}

const GEN_MODEL_NAME = {
    "latent_diffusion": "Latent Diffusion",
    "standard_diffusion": "Standard Diffusion",
    "human": "Human"
}

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

    postImage(PRED_URL, predFormData)
        .then((response) => response.json())
        .then((data) => {
            // Removing Loader
            pred_loading_element.style.display = "none"
            upload_btn_element.style.display = "block"

            pred_res_card_element.style.display = "flex"
            heatmap_image_element.src = "data:image/jpeg;base64," + data.hm_img

            let pred_element_index=0
            for(const class_name in data.prediction_results) {
                pred_display_label_elements.at(pred_element_index).innerHTML = CLASSES_DISP_NAME[class_name]

                let pred_result = Math.floor(data.prediction_results[class_name]*100)
                pred_display_elements.at(pred_element_index).ariaValueNow = pred_result.toString()
                pred_display_fill_elements.at(pred_element_index).innerHTML = pred_result + "%"
                pred_display_fill_elements.at(pred_element_index).style.width = pred_result + "%"

                pred_element_index++

                if (pred_element_index >= 3){
                    break
                }
            }

            let attr_element_index=0
            for(const model_name in data.attribution_scores) {
                attr_display_label_elements.at(attr_element_index).innerHTML = GEN_MODEL_NAME[model_name]

                let attr_result = Math.floor(data.attribution_scores[model_name]*100)
                attr_display_elements.at(attr_element_index).ariaValueNow = attr_result.toString()
                attr_display_fill_elements.at(attr_element_index).innerHTML = attr_result + "%"
                attr_display_fill_elements.at(attr_element_index).style.width = attr_result + "%"

                attr_element_index++
            }
        }).catch(console.error);
}

//inmg src="data:image/jpeg;base64,{{ hm_img }}"
//https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
