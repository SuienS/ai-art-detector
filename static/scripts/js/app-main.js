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

image_upload.addEventListener("change", showImgPreview)
pred_submit_btn_element.addEventListener("click", sendImage)

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
                pred_display_label_elements.at(pred_element_index).innerHTML = class_name

                let pred_result = Math.round(data.prediction_results[class_name]*100)
                pred_display_elements.at(pred_element_index).ariaValueNow = pred_result
                pred_display_fill_elements.at(pred_element_index).innerHTML = pred_result + "%"
                pred_display_fill_elements.at(pred_element_index).style.width = pred_result + "%"

                pred_element_index++
            }
        }).catch(console.error);
}

//inmg src="data:image/jpeg;base64,{{ hm_img }}"
//https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
