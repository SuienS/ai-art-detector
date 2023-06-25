import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms.functional import to_pil_image

from matplotlib import colormaps
from sklearn.preprocessing import StandardScaler

import warnings

imagenet_weights = models.ConvNeXt_Base_Weights.DEFAULT
preprocess_transforms = imagenet_weights.transforms()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_h, img_w = 224, 224


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# scaler = StandardScaler()
with open('model/test_col_feature_scaler.pkl', 'rb') as sc_f: # TODO: Replace with training scaler
    scaler = pickle.load(sc_f)

print("Scaler mean     : ", scaler.mean_)
print("Scaler Variance : ", scaler.var_)

art_class_labels = [
    'art_nouveau',
    'baroque',
    'expressionism',
    'impressionism',
    'post_impressionism',
    'realism',
    'renaissance',
    'romanticism',
    'surrealism',
    'ukiyo_e']


def get_img_colour_features(image, col_feature_scaler):
    rgb_img_features = np.array(image.resize((1, 1))).squeeze()
    hsv_img_features = np.array(image.convert('HSV').resize((1, 1))).squeeze()
    cmyk_img_features = np.array(image.convert('CMYK').resize((1, 1))).squeeze()[:-1]

    return torch.Tensor(col_feature_scaler.transform(
        [np.concatenate((rgb_img_features, hsv_img_features, cmyk_img_features))]
    ))


def get_model_pred(model, art_img_tensor, art_img_col_features):
    for train_param in model.features.parameters():  # TODO: For all layers of just for the features?
        train_param.requires_grad = True

    gradients = None
    activations = None

    def hook_backward(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output

    def hook_forward(module, args, output):
        nonlocal activations
        activations = output

    # art_vision_model.features[-1][-1]
    #
    #     CNBlock(
    #       (block): Sequential(
    #         (0): Conv2d(1024, 1024, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=1024) | Size [1, 1024, 7, 7]
    #         (1): Permute()
    #         (2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
    #         (3): Linear(in_features=1024, out_features=4096, bias=True)
    #         (4): GELU(approximate='none')
    #         (5): Linear(in_features=4096, out_features=1024, bias=True)
    #         (6): Permute()
    #       )
    #       (stochastic_depth): StochasticDepth(p=0.5, mode=row)
    #     )

    hook_backward = model.features[-1][-1].block[0].register_full_backward_hook(hook_backward, prepend=False)
    hook_forward = model.features[-1][-1].block[0].register_forward_hook(hook_forward, prepend=False)

    model.eval()

    preds = model(art_img_tensor.unsqueeze(0), art_img_col_features)
    pred_index = preds.argmax(dim=1)

    preds[:, pred_index].backward()

    hook_backward.remove()
    hook_forward.remove()

    for train_param in model.features.parameters():
        train_param.requires_grad = False

    return preds, pred_index, gradients, activations


def generate_grad_map(gradients, activations):
    avg_pooled_gradients = torch.mean(
        gradients[0],  # Size [1, 1024, 7, 7]
        dim=[0, 2, 3]
    )

    # Weighting acitvation features (channels) using its related calculated Gradient
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= avg_pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # L2 Normalisation # IMPROVED GRADCAM!!!!! EXPERIMENT MORE
    heatmap = F.normalize(heatmap)

    # relu on top of the heatmap
    heatmap = F.sigmoid(heatmap)

    # Min-max normalization of the heatmap
    heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))

    return heatmap.detach()


def predict_image(art_img, model, col_feature_scaler, hm_opacity=0.3):
    art_img = art_img.resize((img_h, img_h), resample=Image.BICUBIC)
    img_col_features = get_img_colour_features(art_img, col_feature_scaler)
    art_img_tensor = preprocess_transforms(art_img)

    preds, pred_index, gradients, activations = get_model_pred(
        model, art_img_tensor.to(device), img_col_features.to(device)
    )
    preds = F.softmax(preds.detach(), dim=1).numpy()

    heatmap = generate_grad_map(gradients, activations)

    hm_overlay = to_pil_image(heatmap.detach().cpu(), mode='F').resize((img_h, img_h), resample=Image.BICUBIC)

    # Jet Colormap
    col_map = colormaps['YlOrRd']
    hm_overlay = Image.fromarray(
        (255 * col_map(np.asarray(hm_overlay) ** 2)[:, :, :3]).astype(np.uint8)
    )

    super_imposed_img = Image.blend(art_img, hm_overlay, alpha=hm_opacity)

    return preds, pred_index, super_imposed_img
