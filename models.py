import torch
from torch import nn
import torchvision as tv


def VGG16(pretrained=False):
    # Define Model
    net = tv.models.vgg16(pretrained=pretrained, progress=True)
    net.features[0] = torch.nn.Conv2d(5, 64, 3, stride=(1, 1), padding=(1, 1))
    net.classifier[-1] = torch.nn.Linear(4096, 2, bias=True)

    return net


def VGG19_BN(pretrained=False):
    # Define Model
    net = tv.models.vgg19_bn(pretrained=pretrained, progress=True)
    net.features[0] = torch.nn.Conv2d(5, 64, 3, stride=(1, 1), padding=(1, 1))
    net.classifier[-1] = torch.nn.Linear(4096, 2, bias=True)

    return net


def RESNET50(pretrained=False):
    # Define Model
    net = tv.models.resnet50(pretrained=pretrained, progress=True)
    net.fc = torch.nn.Linear(2048, 2, bias=True)

    return net


def RESNET101(pretrained=False):
    # Define Model
    net = tv.models.resnet101(pretrained=pretrained, progress=True)
    net.conv1 = torch.nn.Conv2d(5, 64, 7, stride=(2, 2), padding=(3, 3))
    net.fc = torch.nn.Linear(2048, 2, bias=True)

    return net


def RESNET152(pretrained=False):
    # Define Model
    net = tv.models.resnet152(pretrained=pretrained, progress=True)
    net.conv1 = torch.nn.Conv2d(5, 64, 7, stride=(2, 2), padding=(3, 3))
    net.fc = torch.nn.Linear(2048, 2, bias=True)

    return net


def DENSENET161(pretrained=False):
    # Define Model
    net = tv.models.densenet161(pretrained=pretrained, progress=True)
    net.conv1 = torch.nn.Conv2d(5, 96, 7, stride=(2, 2), padding=(3, 3))
    net.classifier = torch.nn.Linear(2208, 2, bias=True)
    return net


def MOBILENETV2(pretrained=False):
    pretrained = False
    net = tv.models.mobilenet_v2(pretrained=pretrained, progress=True)
    net.features[0][0] = torch.nn.Conv2d(
        5, 32, 3, stride=2, padding=1, bias=False)
    net.classifier[-1] = torch.nn.Linear(1280, 2, bias=True)
    return net


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0):
        super(ConvT, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class unFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 8, 8)


class VAE(nn.Module):
    def __init__(self, n_layers=6, base=16, lf=128, n_channels=5):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            Conv(3, base, 3, stride=2, padding=1),  # 256
            Conv(base, 2 * base, 5, padding=2),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),  # 128
            Conv(2 * base, 2 * base, 5, padding=2),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),  # 64
            Conv(2 * base, 4 * base, 5, padding=2),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 32
            Conv(4 * base, 4 * base, 5, padding=2),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 16
            Conv(4 * base, 4 * base, 5, padding=2),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 8
            Conv(4 * base, 4 * base, 5, padding=2),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 4
            nn.Conv2d(4 * base, 64 * base, 4),
            nn.LeakyReLU()
        )

        # self.encoder_mu = nn.Linear(lf, lf)
        # self.encoder_logvar = nn.Linear(lf, lf)
        self.encoder_mu = nn.Conv2d(64 * base, lf, 1)
        self.encoder_logvar = nn.Conv2d(64 * base, lf, 1)

        self.decoder = nn.Sequential(
            Conv(lf, 64 * base, 1),
            ConvT(64 * base, 4 * base, 4),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvT(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 5, padding=2),
            ConvT(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 5, padding=2),
            ConvT(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 5, padding=2),
            ConvT(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 2 * base, 5, padding=2),
            ConvT(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, 2 * base, 5, padding=2),
            ConvT(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, base, 5, padding=2),
            ConvT(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_fm(nn.Module):
    def __init__(self, n_layers=6, base=16, lf=128, n_channels=5):
        super(VAE_fm, self).__init__()

        self.encoder = nn.Sequential(
            Conv(3, base, 3, stride=2, padding=1),  # 256
            Conv(base, 2 * base, 5, padding=2),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),  # 128
            Conv(2 * base, 2 * base, 5, padding=2),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),  # 64
            Conv(2 * base, 4 * base, 5, padding=2),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 32
            Conv(4 * base, 4 * base, 5, padding=2),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),  # 16
            Conv(4 * base, 4 * base, 5, padding=2),
            nn.Conv2d(4 * base, 64 * base, 8),
            nn.LeakyReLU()
        )

        # self.encoder_mu = nn.Linear(lf, lf)
        # self.encoder_logvar = nn.Linear(lf, lf)
        self.encoder_mu = nn.Conv2d(64 * base, lf, 1)
        self.encoder_logvar = nn.Conv2d(64 * base, lf, 1)

        self.decoder = nn.Sequential(
            Conv(lf, 64 * base, 1),
            ConvT(64 * base, 4 * base, 8),
            Conv(4 * base, 4 * base, 5, padding=2),
            ConvT(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 5, padding=2),
            ConvT(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 2 * base, 5, padding=2),
            ConvT(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, 2 * base, 5, padding=2),
            ConvT(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, base, 5, padding=2),
            ConvT(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar