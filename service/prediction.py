import torch
import torch.nn.functional as F

from constants import FM_HEATMAP_CLASS_COUNT, FM_G_CAM_TYPE, G_CAM_TYPE


class PredictionService:

    def __init__(self, model, last_conv_layer):
        self.model = model
        self.last_conv_layer = last_conv_layer

    def get_model_pred(self, art_img_tensor, hm_type):
        grad_list = []
        act_list = []

        for train_param in self.model.parameters():
            train_param.requires_grad = True

        gradients = None
        activations = None

        def hook_backward(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output

        def hook_forward(module, args, output):
            nonlocal activations
            activations = output

        hook_backward = self.last_conv_layer.register_full_backward_hook(hook_backward, prepend=False)
        hook_forward = self.last_conv_layer.register_forward_hook(hook_forward, prepend=False)

        self.model.eval()

        preds = self.model(art_img_tensor.unsqueeze(0))

        # Sort prediction indices
        sorted_pred_indices = torch.argsort(preds, dim=1, descending=True).squeeze(0)

        if hm_type == FM_G_CAM_TYPE:
            # Iterate through the top prediction indices
            for rank in range(FM_HEATMAP_CLASS_COUNT):
                preds[:, sorted_pred_indices[rank]].backward(retain_graph=True)
                grad_list.append(gradients)
                act_list.append(activations)

            hook_backward.remove()
            hook_forward.remove()

        elif hm_type == G_CAM_TYPE:
            preds[:, sorted_pred_indices[0]].backward()

            grad_list.append(gradients)
            act_list.append(activations)

            hook_backward.remove()
            hook_forward.remove()

        for train_param in self.model.parameters():
            train_param.requires_grad = False

        preds = F.softmax(preds.detach(), dim=1).cpu().squeeze(0)

        return preds.tolist(), sorted_pred_indices.tolist(), grad_list, act_list
