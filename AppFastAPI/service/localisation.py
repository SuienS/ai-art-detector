import torch
import torch.nn.functional as F


class LocalisationService:
    """
    This class contains all the functions required to generate the heatmap to explain the model predictions
    """
    @staticmethod
    def generate_fmgcam(gradients_list, activations_list):
        """
        Fused Multi-class Gradient-weighted Class Activation Map

        :param gradients_list: List of model gradients pertaining to multiple classes
        :param activations_list: List of model activations pertaining to multiple classes
        :return: FM-G-CAM heatmap
        """
        heatmaps = []

        # Iterate through the activation maps related to top n classes
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

        # Concatenation of activation maps based on top n classes
        heatmaps = torch.cat(heatmaps)

        # Filter the heatmap based on the maximum weighted activation along the channel axis
        hm_mask_indices = heatmaps.argmax(dim=0).unsqueeze(0)

        hm_3d_mask = torch.cat([hm_mask_indices for _ in range(heatmaps.size()[0])])

        hm_3d_mask = torch.cat(
            [(hm_3d_mask[index] == (torch.ones_like(hm_3d_mask[index]) * index)).unsqueeze(0) for index in
             range(heatmaps.size()[0])]
        ).long()

        heatmaps *= hm_3d_mask

        # L2 Normalisation of the heatmap soften the differences
        heatmaps = F.normalize(heatmaps)

        # relu on top of the heatmap
        heatmaps = F.relu(heatmaps)

        # Min-max normalization of the heatmap
        heatmaps = (heatmaps - torch.min(heatmaps)) / (torch.max(heatmaps) - torch.min(heatmaps))

        return heatmaps.detach().cpu()

    @staticmethod
    def generate_gcam(gradients, activations):
        """
        Implements Grad-CAM
        https://doi.org/10.1109/ICCV.2017.74

        :param gradients: List of model gradients pertaining to multiple classes
        :param activations: List of model activations pertaining to multiple classes
        :return: Grad-CAM
        """
        avg_pooled_gradients = torch.mean(
            gradients[0],  # Size [1, 1024, 7, 7]
            dim=[0, 2, 3]
        )

        # Weighting activation features (channels) using its related calculated Gradient
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= avg_pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        heatmap = F.relu(heatmap)

        # Min-max normalization of the heatmap
        heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))

        return heatmap.detach()
