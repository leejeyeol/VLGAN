import torch.nn as nn

class encoder64x64(nn.Module):
    def __init__(self, num_in_channels=1, z_size=2, num_filters=64, type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_filters * 8, z_size, 4, 2, 1, bias=False),
        )
        if self.type == 'VAE':
            self.fc_mu = nn.Conv2d(z_size, z_size, 1)
            self.fc_sig = nn.Conv2d(z_size, z_size, 1)

    def forward(self, x):
        if self.type == 'AE' or self.type == 'AAE':
            # AE
            z = self.encoder(x)
            return z
        elif self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar
        else:
            print("autoencoder_type is %s, it is unknown." % self.type)


class decoder64x64(nn.Module):
    def __init__(self, num_in_channels=3, z_size=2, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, num_filters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(num_filters * 2, num_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(num_filters, num_in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x


class discriminator64x64(nn.Module):
    def __init__(self, num_in_channels=1, num_filters=64):
        super(discriminator64x64, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(num_in_channels, num_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(num_filters * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class encoder32x32(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64, type='AE'):
        super().__init__()
        self.type = type
        self.encoder = nn.Sequential(
            nn.Conv2d(num_in_channels, 64, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 4 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * num_filters, z_size, 3, 1, 0),
        )
        self.fc_mu = nn.Conv2d(z_size, z_size, 1)
        self.fc_sig = nn.Conv2d(z_size, z_size, 1)

    def forward(self, x):
        if self.type == 'VAE':
            # VAE
            z_ = self.encoder(x)
            mu = self.fc_mu(z_)
            logvar = self.fc_sig(z_)
            return mu, logvar
        else:
            # AE
            z = self.encoder(x)
            return z


class decoder32x32(nn.Module):
    def __init__(self, num_in_channels=1, z_size=80, num_filters=64):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 256, 5, 1, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 5, 1, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 5, 2, 0),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),

            nn.ConvTranspose2d(num_filters, num_in_channels, 4, 2, 0),
            nn.Tanh()
        )

    def forward(self, z):
        recon_x = self.decoder(z)
        return recon_x


class Discriminator32x32(nn.Module):
    def __init__(self, num_in_channels=1, num_filters=64):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(num_in_channels, 64, 5, 2, 1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 2 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(2 * num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 4 * num_filters, 4, 2, 1),
            nn.BatchNorm2d(4 * num_filters),
            nn.LeakyReLU(0.2),

            nn.Conv2d(4 * num_filters, 1, 3, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        d = self.discriminator(x)
        return d


