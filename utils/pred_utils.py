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
heatmap_class_count = 3

art_class_labels = [
    'AI_LD_art_nouveau',
    'AI_LD_baroque',
    'AI_LD_expressionism',
    'AI_LD_impressionism',
    'AI_LD_post_impressionism',
    'AI_LD_realism',
    'AI_LD_renaissance',
    'AI_LD_romanticism',
    'AI_LD_surrealism',
    'AI_LD_ukiyo-e',
    'AI_SD_art_nouveau',
    'AI_SD_baroque',
    'AI_SD_expressionism',
    'AI_SD_impressionism',
    'AI_SD_post_impressionism',
    'AI_SD_realism',
    'AI_SD_renaissance',
    'AI_SD_romanticism',
    'AI_SD_surrealism',
    'AI_SD_ukiyo-e',
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

diffusion_model_labels = [
    "latent_diffusion",
    "standard_diffusion",
    "human"
]

FM_CAM_TYPE = "FM-CAM"


def get_model_pred(model, art_img_tensor):
    grad_list = []
    act_list = []

    for train_param in model.parameters():
        train_param.requires_grad = True

    gradients = None
    activations = None

    def hook_backward(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output

    def hook_forward(module, args, output):
        nonlocal activations
        activations = output

    hook_backward = model.attention_block.register_full_backward_hook(hook_backward, prepend=False)
    hook_forward = model.attention_block.register_forward_hook(hook_forward, prepend=False)

    model.eval()

    preds = model(art_img_tensor.unsqueeze(0))

    # Sort prediction indices
    sorted_pred_indices = torch.argsort(preds, dim=1, descending=True).squeeze(0)

    # Iterate through the top prediction indices
    for rank in range(heatmap_class_count):
        preds[:, sorted_pred_indices[rank]].backward(retain_graph=True)
        grad_list.append(gradients)
        act_list.append(activations)

    hook_backward.remove()
    hook_forward.remove()

    for train_param in model.parameters():
        train_param.requires_grad = False

    preds = F.softmax(preds.detach(), dim=1).cpu().squeeze(0)

    return preds.tolist(), sorted_pred_indices.tolist(), grad_list, act_list


def generate_grad_fmcam(gradients_list, activations_list):
    heatmaps = []

    for index, activations in enumerate(activations_list):
        gradients = gradients_list[index]

        avg_pooled_gradients = torch.mean(
            gradients[0],  # Size [1, 1024, 7, 7]
            dim=[0, 2, 3]
        )

        # Weighting activation features (channels) using its related calculated Gradient
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= avg_pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        heatmaps.append(heatmap.unsqueeze(0).detach().cpu())

    heatmaps = torch.cat(heatmaps)

    hm_mask_indices = heatmaps.argmax(dim=0).unsqueeze(0)

    hm_3d_mask = torch.cat([hm_mask_indices for _ in range(hm_mask_indices.max() + 1)])

    hm_3d_mask = torch.cat(
        [(hm_3d_mask[index] == (torch.ones_like(hm_3d_mask[index]) * index)).unsqueeze(0) for index in
         range(heatmaps.size()[0])]
    ).long()

    heatmaps *= hm_3d_mask
    heatmaps = F.normalize(heatmaps)

    # relu on top of the heatmap
    heatmaps = F.relu(heatmaps)

    # Min-max normalization of the heatmap
    heatmaps = (heatmaps - torch.min(heatmaps)) / (torch.max(heatmaps) - torch.min(heatmaps))

    return heatmaps.detach().cpu()


def get_model_pred_gcam(model, art_img_tensor):
    for train_param in model.parameters():
        train_param.requires_grad = True

    gradients = None
    activations = None

    def hook_backward(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output

    def hook_forward(module, args, output):
        nonlocal activations
        activations = output

    hook_backward = model.attention_block.register_full_backward_hook(hook_backward, prepend=False)
    hook_forward = model.attention_block.register_forward_hook(hook_forward, prepend=False)

    model.eval()

    preds = model(art_img_tensor.unsqueeze(0))

    # Sort prediction indices
    sorted_pred_indices = torch.argsort(preds, dim=1, descending=True).squeeze(0)

    preds[:, sorted_pred_indices[0]].backward()

    hook_backward.remove()
    hook_forward.remove()

    for train_param in model.parameters():
        train_param.requires_grad = False

    preds = F.softmax(preds.detach(), dim=1).cpu().squeeze(0)
    print(gradients)
    return preds.tolist(), sorted_pred_indices.tolist(), gradients, activations


def generate_grad_map_gcam(gradients, activations):
    avg_pooled_gradients = torch.mean(
        gradients[0],  # Size [1, 1024, 7, 7]
        dim=[0, 2, 3]
    )

    # Weighting acitvation features (channels) using its related calculated Gradient
    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= avg_pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    #    heatmap = F.normalize(heatmap)

    # relu on top of the heatmap
    heatmap = F.relu(heatmap)

    # Min-max normalization of the heatmap
    heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))

    return heatmap.detach()


def predict_image(art_img, hm_type, model, hm_opacity=0.3) -> (list, int, Image):
    art_img = art_img.resize((img_h, img_h), resample=Image.BICUBIC)
    art_img_tensor = preprocess_transforms(art_img)
    if hm_type == FM_CAM_TYPE:
        preds, sorted_pred_index, gradients, activations = get_model_pred(model, art_img_tensor.to(device))
        heatmaps = generate_grad_fmcam(gradients, activations)

        hm_overlay = to_pil_image(heatmaps, mode='RGB').resize((img_h, img_h), resample=Image.BICUBIC)
    else:
        preds, sorted_pred_index, gradients, activations = get_model_pred_gcam(model, art_img_tensor.to(device))
        heatmap = generate_grad_map_gcam(gradients, activations)

        hm_overlay = to_pil_image(heatmap.detach().cpu(), mode='F').resize((img_h, img_h), resample=Image.BICUBIC)

        # Jet Colormap
        col_map = colormaps['YlOrRd']
        hm_overlay = Image.fromarray(
            (255 * col_map(np.asarray(hm_overlay) ** 2)[:, :, :3]).astype(np.uint8)
        )

    super_impossed_img = Image.blend(art_img, hm_overlay, alpha=hm_opacity)

    attribution_scores = get_attribution_scores(preds)

    return preds, attribution_scores, sorted_pred_index, super_impossed_img


def get_attribution_scores(preds):
    ld_score = np.sum(preds[0:10])
    sd_score = np.sum(preds[10:20])
    real_score = np.sum(preds[20:])

    attr_preds = [ld_score, sd_score, real_score]

    return attr_preds
