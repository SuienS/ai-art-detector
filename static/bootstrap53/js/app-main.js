const PRED_URL = "/prediction/art"
const image_upload = document.getElementById("image-file-input")
const pred_form_element = document.getElementById("pred-form-data");
const pred_btn_element = document.getElementById("pred-submit");
const heatmap_image_element = document.getElementById("heatmap-image");

image_upload.addEventListener("change", showPreview)
pred_btn_element.addEventListener("click", sendImage)

function showPreview() {
    if (image_upload.files.length > 0) {
        var src = URL.createObjectURL(image_upload.files[0]);
        var preview = document.getElementById("img-upload-file");
        preview.src = src;
        preview.style.display = "block";
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

function base64encode(str) {
    return btoa(unescape(encodeURIComponent(str)));
}

function sendImage() {

    const predFormData = new FormData(pred_form_element);
    for (var pair of predFormData.entries()) {
        console.log(pair[0] + ', ' + pair[1]);
    }
    postImage(PRED_URL, predFormData).then((response) => response.json())
        .then((data) => {
            console.log(data.preds)
            heatmap_image_element.src = "data:image/jpeg;base64,"+data.hm_img
        }).catch(console.error);
}

//inmg src="data:image/jpeg;base64,{{ hm_img }}"
//https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
