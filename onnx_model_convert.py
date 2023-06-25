import torch
import tensorflow as tf
import tensorflowjs as tfjs

from utils.pred_utils import art_class_labels, device
from model.model import ArtVisionModel
from model.model_base import ArtVisionModelBase

import onnx
from onnx_tf.backend import prepare


# Load model

# art_vision_model = torch.load("./model/art_brain_model_scripted.pt", map_location=device)


def main():
    onnx_model_path = './js_app/model/art_brain_onnx_model.onnx'
    tf_model_path = './js_app/model/art_brain_tf_model.pb'
    tf_h5_model_path = './js_app/model/art_brain_tf_model.pb'
    tf_json_model_path = '/js_app/model/art_brain_tf_model.json'

    print("[INFO]: Loading model...")
    art_vision_model = ArtVisionModelBase(len(art_class_labels)).to(device)
    # art_vision_model.load_state_dict(torch.load("./js_app/model/art_classifier.pt", map_location=device))
    art_vision_model.eval()

    dummy_input = torch.zeros(1, 3, 224, 224)#, torch.zeros(1, 1, 9)

    print("[INFO]: Exporting ONNX model...")
    torch.onnx.export(
        art_vision_model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        './js_app/model/art_brain_onnx_model.onnx',  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=9,  # the ONNX version to export the model to
        do_constant_folding=False,
        verbose=True
    )

    print("[INFO]: Checking ONNX model...")
    onnx_model = onnx.load(onnx_model_path)  # load onnx model
    onnx.checker.check_model(onnx_model)
    print("[INFO]: Checking ONNX model verified!")

    print("[INFO]: Converting to tensorflow...")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    print("[INFO]: Conversion Complete!")


    # tf_model = tf.keras.models.load_model(tf_model_path)
    #
    # tf_model.save(tf_h5_model_path)
    #
    # #tf_model = tf.keras.models.load_model(tf_h5_model_path)
    # tfjs.converters.save_keras_model(tf_model, tf_json_model_path)
    # # tf_model = onnx_tf.convert_from_onnx(onnx_model)
    # # #tf_rep = prepare(onnx_model)  # prepare tf representation
    # # #tf_rep.export_graph(tf_model_path)  # export the model
    # # tf.io.write_graph(tf_model, "/js_app/model", "art_brain_tf_model.pb", as_text=False)


if __name__ == '__main__':
    main()
