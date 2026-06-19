import torch
import torch.nn as nn
import timm


# Custom AstraNetBT Model
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.out = nn.Linear(dim * 2, dim)

    def forward(self, x_cnn, x_trans):
        combined = torch.cat([x_cnn, x_trans], dim=1)
        g = self.gate(combined)
        fused = g * x_cnn + (1 - g) * x_trans
        return fused

class AstraNetBT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.cnn_backbone = timm.create_model('convnext_tiny',
                                              pretrained=True,
                                              num_classes=0,
                                              global_pool='avg')
        self.trans_backbone = timm.create_model('swin_tiny_patch4_window7_224',
                                                pretrained=True,
                                                num_classes=0,
                                                global_pool='avg')
        self._freeze_early_layers(self.cnn_backbone, freeze_fraction=0.5)
        self._freeze_early_layers(self.trans_backbone, freeze_fraction=0.5)
        cnn_dim = 768
        trans_dim = 768
        self.fusion = GatedFusion(cnn_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(cnn_dim),
            nn.Dropout(0.5),
            nn.Linear(cnn_dim, num_classes)
        )

    def _freeze_early_layers(self,
                             model,
                             freeze_fraction=0.5):
        total_params = len(list(model.parameters()))
        freeze_up_to = int(total_params * freeze_fraction)
        for i, param in enumerate(model.parameters()):
            if i < freeze_up_to:
                param.requires_grad = False

    def forward(self, x):
        feat_cnn = self.cnn_backbone(x)
        feat_trans = self.trans_backbone(x)

        fused = self.fusion(feat_cnn, feat_trans)
        return self.head(fused)
def get_model(num_classes):
    return AstraNetBT(num_classes)
