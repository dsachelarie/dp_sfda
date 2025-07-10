import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys
from nets.custom_wide_resnet import WideResNet
from nets.custom_pretrained_resnet50 import ResNet50
from nets.resnet_gn import resnet50


class WSConv2d(nn.Conv2d):
    """Weight-Standardized Conv2D."""
    def forward(self, x):
        # Standardize weights
        mean = self.weight.mean(dim=(1, 2, 3), keepdim=True)
        std = self.weight.std(dim=(1, 2, 3), keepdim=True) + 1e-5  # Add epsilon to avoid division by zero
        weight_standardized = (self.weight - mean) / std
        return F.conv2d(x, weight_standardized, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def convert_to_wsconv(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            wsconv = WSConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None),
            )
            wsconv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                wsconv.bias.data.copy_(module.bias.data)
            setattr(model, name, wsconv)
        else:
            convert_to_wsconv(module)
    return model



class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.g = nn.Parameter(torch.ones(out_features, 1))
        self.v = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_normal_(self.v)

    def forward(self, x):
        weight = self.g * self.v / (self.v.norm(dim=1, keepdim=True) + 1e-6)
        return torch.nn.functional.linear(x, weight)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class ShotModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(ShotModel, self).__init__()
        if backbone == "alexnet":
            # self.backbone = nn.Sequential(  # TODO review architecture
            #     nn.InstanceNorm2d(3),
            #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #     nn.ReLU(inplace=True),
            #     nn.MaxPool2d(kernel_size=2, stride=2)
            # )
            self.backbone = nn.Sequential(
                nn.InstanceNorm2d(3),
                nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            # self.backbone = WideResNet(40, 31, 4)
            self.bottleneck = nn.Sequential(
                # nn.Linear(256 * 4 * 4, 256),
                nn.Linear(256 * 3 * 3, 256),
                # nn.Linear(128, 256),
                # nn.BatchNorm1d(256, affine=True),
                # nn.Dropout(p=0.5)
                nn.GroupNorm(16, 256)
            )

            self.bottleneck[0].apply(init_weights)

            self.classifier = WeightNormLinear(256, num_classes)

        elif backbone == "lenet":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )

            self.bottleneck = nn.Sequential(
                nn.Linear(50 * 4 * 4, 256),
                # nn.Linear(50 * 4 * 4, 256),
                nn.GroupNorm(16, 256),
                # nn.BatchNorm1d(256, affine=True),
                # nn.Dropout(p=0.5)
            )

            self.bottleneck[0].apply(init_weights)

            self.classifier = WeightNormLinear(256, num_classes)

        elif backbone == "dtn":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(16, 64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(16, 128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(16, 256),
                nn.Dropout2d(0.5),
                nn.ReLU()
            )

            self.bottleneck = nn.Sequential(
                nn.Linear(256 * 4 * 4, 256),
                # nn.Linear(50 * 4 * 4, 256),
                nn.GroupNorm(16, 256),
                # nn.BatchNorm1d(256, affine=True),
                # nn.Dropout(p=0.5)
            )

            self.bottleneck[0].apply(init_weights)

            self.classifier = WeightNormLinear(256, num_classes)

        elif backbone == "wideresnet":
            self.backbone = WideResNet(40, num_classes, 4)

            self.bottleneck = nn.Identity()
            # self.bottleneck = nn.Sequential(
            #     nn.Linear(256, 256),
            #     nn.GroupNorm(16, 256)
            # )

            # self.bottleneck[0].apply(init_weights)

            self.classifier = WeightNormLinear(256, num_classes)
            # self.classifier = nn.Linear(256, num_classes)

        elif backbone == "dirt-t_cnn":
            self.backbone = nn.Sequential(
                # norm(3),
                nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),
                GaussianNoise(sigma=1.0),

                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
                nn.Dropout(0.5),
                GaussianNoise(sigma=1.0),

                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
                nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )

        elif backbone == "resnet50":
            self.backbone = resnet50()

            state_dict = torch.load("weights/ImageNet-ResNet50-GN.pth")["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            self.backbone.load_state_dict(state_dict)

            # self.backbone = convert_to_wsconv(self.backbone)

            self.bottleneck = nn.Sequential(
                nn.Linear(1000, 256),
                nn.GroupNorm(16, 256)
                # nn.BatchNorm1d(256, affine=True),
                # nn.Dropout(p=0.5)
            )

            self.bottleneck[0].apply(init_weights)

            self.classifier = WeightNormLinear(256, num_classes)

        # self.backbone = models.vgg16(pretrained=True).features
        # self.backbone = WideResNet(40, 31, 4)
        # self.backbone = models.alexnet(pretrained=False).features
        # self.backbone = models.resnet50(pretrained=True).features
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-1])

        # self.bottleneck = nn.Sequential(
        #     # nn.Linear(2048, 256),
        #     # nn.Linear(512 * 7 * 7, 256),
        #     # nn.Linear(256 * 6 * 6, 256),
        #     nn.Linear(256 * 3 * 3, 256),
        #     # nn.BatchNorm1d(256)
        #     nn.GroupNorm(16, 256) # DP, instead of BatchNorm
        # )
        # self.bottleneck[0].apply(init_weights)

        # self.classifier = nn.utils.weight_norm(nn.Linear(256, num_classes)) # weight_norm is DP-compatible
        # self.classifier = nn.Linear(256, num_classes)
        # self.classifier = WeightNormLinear(256, num_classes)
        # Apply weight initialization manually since we don't have bias
        # nn.init.xavier_normal_(self.classifier.v)
        # self.classifier.apply(init_weights)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        # print(x.shape, file=sys.stderr)
        x = self.bottleneck(x)
        x = self.classifier(x)
        return x
