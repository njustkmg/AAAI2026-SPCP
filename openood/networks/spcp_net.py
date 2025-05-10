import torch
import torch.nn as nn
import torch.nn.functional as F


class SPCPNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(SPCPNet, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.register_buffer('th', torch.tensor([1000.0]))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_p(self, p):
        self.p = p

    def set_T(self, T):
        self.th = torch.tensor([T]).cuda()

    def forward(self, x, return_feature=False):
        _, feature = self.backbone(x, return_feature=True)

        contribution_matrix = feature.unsqueeze(-1) * self.backbone.fc.weight.T

        B, D, C = contribution_matrix.shape
        if self.num_classes >= 200: # for imagenet
            sample_batch = 16 
            idx = torch.randperm(B, device=contribution_matrix.device)[:sample_batch]
            sampled_matrix = contribution_matrix[idx]
            B = sample_batch
        else:
            sampled_matrix = contribution_matrix

        quantile_val = torch.mean(torch.quantile(sampled_matrix.reshape(B, -1), q=self.p, dim=-1, keepdim=True)).item()
        print(self.th)
        self.th = self.alpha * self.th + (1 - self.alpha) * quantile_val

        logits_cls = torch.sum(torch.clamp(contribution_matrix, max=self.th),dim=1) + self.backbone.fc.bias

        return logits_cls

    def forward_ood_inference(self, x, return_feature=False):
        _, feature = self.backbone(x, return_feature=True)
        contribution_matrix = feature.unsqueeze(-1) * self.backbone.fc.weight.T
        logits_cls = torch.sum(torch.clamp(contribution_matrix, max=self.th),dim=1) + self.backbone.fc.bias

        if return_feature:
            return logits_cls, feature
        return logits_cls

    def forward_threshold(self, x, threshold):
        _, feature = self.backbone(x, return_feature=True)
        feature = feature.clip(max=threshold)
        logits_cls = torch.sum(torch.clamp(contribution_matrix, max=self.th),dim=1) + self.backbone.fc.bias
        return logits_cls

    def get_fc(self):
        fc = self.backbone.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.backbone.fc