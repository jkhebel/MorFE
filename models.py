import torch
from torch import nn
import torch.nn.functional as F
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
    net.conv1 = torch.nn.Conv2d(5, 64, 7, stride=(2, 2), padding=(3, 3))
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
    net
    return net

# https://becominghuman.ai/variational-autoencoders-for-new-fruits-with-keras-and-pytorch-6d0cfc4eeabd


# https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html#vae, added center crop function


class VAE2(nn.Module):
    def __init__(self, in_shape, n_classes, n_latent, layers=2, bf=32):
        super().__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c, h, w = in_shape
        self.h_dim = h // 2**layers  # receptive field downsampled 2 times
        self.w_dim = w // 2**layers  # receptive field downsampled 2 times
        self.layers = layers
        self.bf = bf

        enc_layers = [nn.BatchNorm2d(c)]
        for layer in range(layers):
            i = c if (layer == 0) else (bf * (layer))
            o = bf * (layer + 1)
            enc_layers.extend([
                nn.Conv2d(i, o, kernel_size=4, stride=2,
                          padding=1),  # 32, 16, 16
                nn.BatchNorm2d(o),
                nn.LeakyReLU()
            ])
        self.encoder = nn.Sequential(*enc_layers)

        self.z_mean = nn.Linear(
            layers * bf * self.h_dim * self.w_dim, n_latent)
        self.z_var = nn.Linear(layers * bf * self.h_dim * self.w_dim, n_latent)
        self.z_develop = nn.Linear(
            n_latent, layers * bf * self.h_dim * self.w_dim)

        dec_layers = []
        for layer in reversed(range(1, layers)):
            i = bf * (layer + 1)
            o = bf * layer
            dec_layers.extend([
                nn.ConvTranspose2d(i, o, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm2d(o),
                nn.ReLU(),
            ])
        dec_layers.extend([
            nn.ConvTranspose2d(bf, n_classes, kernel_size=3,
                               stride=2, padding=1)
        ])
        self.decoder = nn.Sequential(*dec_layers)

    def center_crop(self, img, h, w):
        crop_h = torch.FloatTensor([img.size()[2]]).sub(h).div(-2)
        crop_w = torch.FloatTensor([img.size()[3]]).sub(w).div(-2)

        return F.pad(img, [
            crop_w.ceil().int()[0], crop_w.floor().int()[0],
            crop_h.ceil().int()[0], crop_h.floor().int()[0],
        ])

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = stddev.new_tensor(torch.randn(stddev.size()))
        return (noise * stddev) + mean

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        _, h, w = self.in_shape
        out = self.z_develop(z)
        out = out.view(z.size(0), self.layers *
                       self.bf, self.h_dim, self.w_dim)
        out = self.decoder(out)
        out = self.center_crop(out, h, w)
        out = nn.Sigmoid()(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar


# class Autoencoder(nn.Model):
#     def __init__(self, in_channels, out_channels, bf=16):
#         super().__init__()
#         self.encoder = nn.Sequential(  # 5x512x512
#             nn.Conv2d(in_channels, bf, k, s, p)
#             nn.Conv2d(in_channels, 16, 3, stride=3,
#                       padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, out_channels, 2, stride=2,
#                                padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        base = 16

        self.encoder = nn.Sequential(
            Conv(5, base, 3, stride=2, padding=1),
            Conv(base, 2 * base, 3, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, padding=1),
            Conv(2 * base, 2 * base, 3, stride=2, padding=1),
            Conv(2 * base, 4 * base, 3, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, padding=1),
            Conv(4 * base, 4 * base, 3, stride=2, padding=1),
            nn.Conv2d(4 * base, 64 * base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64 * base, 32 * base, 1)
        self.encoder_logvar = nn.Conv2d(64 * base, 32 * base, 1)

        self.decoder = nn.Sequential(
            nn.Conv2d(32 * base, 64 * base, 1),
            ConvTranspose(64 * base, 4 * base, 8),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvTranspose(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 4 * base, 3, padding=1),
            ConvTranspose(4 * base, 4 * base, 4, stride=2, padding=1),
            Conv(4 * base, 2 * base, 3, padding=1),
            ConvTranspose(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, 2 * base, 3, padding=1),
            ConvTranspose(2 * base, 2 * base, 4, stride=2, padding=1),
            Conv(2 * base, base, 3, padding=1),
            ConvTranspose(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 5, 3, padding=1),
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
