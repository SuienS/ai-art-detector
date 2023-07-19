import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from matplotlib import colormaps

from constants import FM_CAM_TYPE, G_CAM_TYPE, IMG_H, IMG_W
from service.localisation import generate_fmgcam, generate_gcam
from service.prediction import get_model_pred
from utils.pred_utils import get_attribution_scores


class InferencingService:
    def __init__(self, model, imagenet_weights, preprocess_transforms, device):
        self.model = model
        self.imagenet_weights = imagenet_weights
        self.preprocess_transforms = preprocess_transforms
        self.device = device

    def predict_image(self, art_img, hm_type, hm_opacity=None) -> (list, int, Image):
        global preds, hm_overlay, sorted_pred_index

        art_img = art_img.resize((IMG_H, IMG_W), resample=Image.BICUBIC)
        art_img_tensor = self.preprocess_transforms(art_img)
        # import matplotlib.pyplot as plt
        # plt.imshow(art_img_tensor.squeeze().permute(1, 2, 0), cmap="gray")
        # plt.savefig('./processed_image.jpg')

        if hm_type == FM_CAM_TYPE:
            preds, sorted_pred_index, grad_list, act_list = get_model_pred(self.model,
                                                                           art_img_tensor.to(self.device),
                                                                           FM_CAM_TYPE)
            heatmaps = generate_fmgcam(grad_list, act_list)

            hm_overlay = to_pil_image(heatmaps, mode='RGB').resize((IMG_H, IMG_H), resample=Image.BICUBIC)

        elif hm_type == G_CAM_TYPE:
            preds, sorted_pred_index, gradients, activations = get_model_pred(self.model,
                                                                              art_img_tensor.to(self.device),
                                                                              G_CAM_TYPE)
            heatmap = generate_gcam(gradients[0], activations[0])

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
