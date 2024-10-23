"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import BatchNorm2d

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=0, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = self.conv_atten(feat)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(feat, atten)
        return out
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class FFM(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FFM, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        self.conv = ConvBNReLU(out_chan, out_chan, ks=1)
        #  self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x ,y):
        x = x.unsqueeze(-1).unsqueeze(-1)
        y = y.unsqueeze(-1).unsqueeze(-1)
        features = x + y
        # features = features.unsqueeze(-1).unsqueeze(-1)
        atten = self.conv_atten(features)
        atten = self.bn_atten(atten)
        #  atten = self.sigmoid_atten(atten)
        atten = atten.sigmoid()
        out = torch.mul(x, atten)+ torch.mul(y, 1-atten)
        out = self.conv(out)
        return out
    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Dinov2(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, num_classes: int = 1000,head_init_scale: float = 1.):
        super().__init__()
        self.dinov2_vitt14 = torch.hub.load('/home/bch/code/dinov2', 'dinov2_vitb14',source='local')
        # self.att111 = AttentionRefinementModule(768*5,768)
        self.att222 = FFM(768,768)
        self.head = nn.Linear(768, num_classes)
        self.head.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # features_dict = self.dinov2_vitt14.forward_features(x)
        # # features = features_dict['x_norm_regtokens']
        # cls_token = features_dict["x_norm_clstoken"]
        # patch_tokens = features_dict["x_norm_patchtokens"]
        
        #     # fmt: off
        # features = torch.cat([
        #         cls_token,
        #         patch_tokens.mean(dim=1),
        #     ], dim=1)
        # # features = features.unsqueeze(-1).unsqueeze(-1)
        # return features
        # x = self.dinov2_vitb14.get_intermediate_layers(x, n=4, return_class_token=True)
        #     # fmt: off
        # features = torch.cat([
        #         x[0][1],
        #         x[1][1],
        #         x[2][1],
        #         x[3][1],
        #         x[3][0].mean(dim=1),
        # ], dim=1)
        # features = features.unsqueeze(-1).unsqueeze(-1)
        # return features  # global average pooling, (N, C, H, W) -> (N, C)
        
        features_dict = self.dinov2_vitt14.forward_features(x)
        cls_token = features_dict["x_norm_clstoken"]
        patch_tokens = features_dict["x_norm_patchtokens"].mean(dim=1)
        return cls_token,patch_tokens
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token,patch_tokens = self.forward_features(x)
        # x = self.forward_features(x)
        x = self.att222(cls_token,patch_tokens)
        x = x.squeeze()
        # print(x.shape)
        x = self.head(x)
        return x


def dinov2_tiny(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = Dinov2(num_classes=num_classes)
    return model

if  __name__ == "__main__":
    model = dinov2_tiny(5)
    # 将模型移动到适当的设备（如GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 打印模型的摘要信息，包括参数量和模型结构
    summary(model, input_size=(3, 518, 518))  # 假设输入张量的形状是 (3, 224, 224)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total trainable parameters: {:,}".format(total_params))