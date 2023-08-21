import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from matplotlib import colormaps

from utils.constants import FM_G_CAM_TYPE, G_CAM_TYPE, IMG_H, IMG_W
from service.localisation import LocalisationService
from service.prediction import PredictionService
from utils.pred_utils import get_attribution_scores


class InferencingService:
    """
    This class contains all the functions needed for the inferencing
    """
    def __init__(self, model, preprocess_transforms, device):
        self.model = model
        self.preprocess_transforms = preprocess_transforms
        self.device = device
        self.prediction_service = PredictionService(
            model=model,
            last_conv_layer=model.attention_module
        )
        self.localisation_service = LocalisationService()

    def predict_image(self, art_img, hm_type, hm_opacity=None) -> (list, int, Image):
        """
        This function executes the main prediction mechanism of the system.

        :param art_img: Input artistic image
        :param hm_type: Heatmap Type - G-CAM or FM-G-CAM
        :param hm_opacity: Opacity of the heatmap

        :return: Prediction results, Attribution scores, Uploaded image and Heatmap
        """
        preds, hm_overlay, sorted_pred_index = None, None, None

        art_img = art_img.resize((IMG_H, IMG_W), resample=Image.BICUBIC).convert('RGB')
        art_img_tensor = self.preprocess_transforms(art_img)
        # Uncomment below to save image
        # import matplotlib.pyplot as plt
        # plt.imshow(art_img_tensor.squeeze().permute(1, 2, 0), cmap="gray")
        # plt.savefig('./processed_image.jpg')

        if hm_type == FM_G_CAM_TYPE:
            preds, sorted_pred_index, grad_list, act_list = self.prediction_service.get_model_pred(
                art_img_tensor.to(self.device),
                FM_G_CAM_TYPE
            )
            heatmaps = self.localisation_service.generate_fmgcam(grad_list, act_list)

            hm_overlay = to_pil_image(heatmaps, mode='RGB').resize((IMG_H, IMG_H), resample=Image.BICUBIC)

        elif hm_type == G_CAM_TYPE:
            preds, sorted_pred_index, gradients, activations = self.prediction_service.get_model_pred(
                art_img_tensor.to(self.device),
                G_CAM_TYPE)
            heatmap = self.localisation_service.generate_gcam(gradients[0], activations[0])

            hm_overlay = to_pil_image(heatmap.detach().cpu(), mode='F').resize((IMG_H, IMG_H), resample=Image.BICUBIC)

            # Jet Colormap
            col_map = colormaps['jet']
            hm_overlay = Image.fromarray(
                (255 * col_map(np.asarray(hm_overlay) ** 2)[:, :, :3]).astype(np.uint8)
            )

        attribution_scores = get_attribution_scores(preds)

        if hm_opacity:
            super_imposed_img = Image.blend(art_img, hm_overlay, alpha=hm_opacity)
            return preds, attribution_scores, sorted_pred_index, super_imposed_img
        else:
            return preds, attribution_scores, sorted_pred_index, art_img, hm_overlay
