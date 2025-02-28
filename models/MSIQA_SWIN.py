import torch.nn as nn
import timm


class SwinTransformerNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SwinTransformerNet, self).__init__()

        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=pretrained)
        self.model.head = nn.Identity()


    def forward(self, x):
        # save the outpus of intermediate stages
        outputs = {}

        # Patch Embedding
        x = self.model.patch_embed(x)

        # Intermediate stages
        for i, layer in enumerate(self.model.layers):
            x = layer(x)
            outputs[f'Stage{i + 1}'] = x

        return outputs