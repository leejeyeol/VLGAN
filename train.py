import torch.utils.data as ud
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import glob as glob
from PIL import Image
import model

import matplotlib.pyplot as plt

import utils


time_calc = utils.Time_calculator()


# =======================================================================================================================
# Options
# =======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataset', default='CelebA', help='what is dataset?',
                    choices=['CelebA',  'MNIST'])
parser.add_argument('--dataroot',
                    default='/home/mlpa/Workspace/dataset/CelebA/Img/img_anlign_celeba_png.7z/img_align_celeba_png',
                    help='path to dataset.')
parser.add_argument('--pretrainedEpoch', type=int, default=0,
                    help="path of Decoder networks. '0' is training from scratch.")
parser.add_argument('--modelOutFolder', default='/home/mlpa/Workspace/experimental_result/LJY/VAEGAN',
                    help="folder to model checkpoints.")
parser.add_argument('--resultOutFolder', default='/home/mlpa/data_4T/experiment_results/ljy/results',
                    help="folder to test results")
parser.add_argument('--save_tick', type=int, default=1, help='save tick. default is 1')
parser.add_argument('--epoch', type=int, default=255, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--model', type=str, default='pretrained_model', help='Model name')
parser.add_argument('--nc', type=int, default=1, help='number of input channel.')
parser.add_argument('--nz', type=int, default=100, help='number of input channel.')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters.')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters.')

parser.add_argument('--seed', type=int, help='manual seed')

parser.add_argument('--TrainingTimesave', action='store_true', help='save csv for gradient')
parser.add_argument('--save', action='store_true', help='save options. default:False.')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--display', action='store_true', help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--display_type', default='per_iter', help='displat tick', choices=['per_epoch', 'per_iter'])


options = parser.parse_args()

print(options)

visualize_latent = False
recon_learn = True
cycle_learn = False
recon_weight = 5
encoder_weight = 1.0
decoder_weight = 1.0

options.autoencoderType = 'VAE'
options.pretrainedModelName = options.preset + '_' + options.dataset


# criterion set
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
L1_loss = nn.L1Loss()


def Variational_loss(input, target, mu, logvar):
    recon_term_weight = 1
    kld_term_weight = 1
    recon_loss = L1_loss(input, target)
    batch_size = logvar.data.shape[0]
    nz = logvar.data.shape[1]
    KLD_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (nz * batch_size)
    return recon_term_weight * recon_loss, kld_term_weight * KLD_loss


class custom_Dataloader(torch.utils.data.Dataset):
    def __init__(self, path, transform, type='train'):
        super().__init__()
        self.type = type
        self.transform = transform

        assert os.path.exists(path)
        self.base_path = path

        cur_file_paths = glob.glob(self.base_path + '/*.png')
        cur_file_paths.sort()

        self.file_paths, self.val_paths, self.test_paths = utils.tvt_divider(cur_file_paths, train_ratio=4,
                                                                                 val_ratio=1, test_ratio=1)

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __len__(self):
        if self.type == 'train':
            return len(self.file_paths)
        elif self.type == 'val':
            return len(self.val_paths)
        elif self.type == 'test':
            return len(self.test_paths)

    def __getitem__(self, item):
        if self.type == 'train':
            path = self.file_paths[item]
        elif self.type == 'val':
            path = self.val_paths[item]
        elif self.type == 'test':
            path = self.test_paths[item]
        img = self.pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, 1



# save directory make   ================================================================================================
try:
    os.makedirs(options.modelOutFolder)
except OSError:
    pass

# seed set  ============================================================================================================
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
random.seed(options.seed)
torch.manual_seed(options.seed)

# cuda set  ============================================================================================================
if options.cuda:
    torch.cuda.manual_seed(options.seed)

torch.backends.cudnn.benchmark = True
cudnn.benchmark = True
if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def weight_init(module):
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_normal(module.weight.data)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal(module.weight.data)

    nz = int(options.nz)
    if options.dataset == 'MNIST':
        encoder = model.encoder32x32(num_in_channels=1, z_size=nz, num_filters=64, type=autoencoder_type)
        encoder.apply(LJY_utils.weights_init)
        print(encoder)

        decoder = model.decoder32x32(num_in_channels=1, z_size=nz, num_filters=64)
        decoder.apply(LJY_utils.weights_init)
        print(decoder)

        discriminator = model.Discriminator32x32(num_in_channels=1, num_filters=64)
        discriminator.apply(LJY_utils.weights_init)
        print(discriminator)

    elif options.dataset == 'CelebA':
            encoder = model.encoder64x64(num_in_channels=3, z_size=nz, num_filters=64, type=autoencoder_type)
            encoder.apply(LJY_utils.weights_init)
            print(encoder)

            decoder = model.decoder64x64(num_in_channels=3, z_size=nz, num_filters=64)
            decoder.apply(LJY_utils.weights_init)
            print(decoder)

            discriminator = model.discriminator64x64(num_in_channels=3, num_filters=64)
            discriminator.apply(LJY_utils.weights_init)
            print(discriminator)

    if options.cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        MSE_loss.cuda()
        BCE_loss.cuda()
        L1_loss.cuda()

    return encoder, decoder, discriminator, z_discriminator


# =======================================================================================================================
# Data and Parameters
# =======================================================================================================================

# MNIST call and load   ================================================================================================
autoencoder_type = options.autoencoderType
nz = options.nz
ngpu = options.ngpu
encoder, decoder, discriminator, z_discriminator = model_init(autoencoder_type)

# =======================================================================================================================
# Training
# =======================================================================================================================


# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizerD = optim.Adam(decoder.parameters(), betas=(0.5, 0.9), lr=0.0005)
optimizerE = optim.Adam(encoder.parameters(), betas=(0.5, 0.9), lr=0.0001)
optimizerDiscriminator = optim.Adam(discriminator.parameters(), betas=(0.5, 0.9), lr=0.0005)

# training start
def train():
    validation_path = os.path.join(os.path.dirname(options.modelOutFolder),
                                   '%s_%s_%s' % (options.dataset, options.intergrationType, options.autoencoderType))
    validation_path = LJY_utils.make_dir(validation_path, allow_duplication=True)
    save_path = os.path.join(options.modelOutFolder)
    save_path = LJY_utils.make_dir(save_path)
    ep = options.pretrainedEpoch
    if ep != 0:
        if options.DiscriminatorLoad is True:
            discriminator.load_state_dict(torch.load(
                os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % ep)))
        else:
            encoder.load_state_dict(torch.load(
                os.path.join(options.modelOutFolder, options.pretrainedModelName + "_encoder" + "_%d.pth" % ep)))
            decoder.load_state_dict(torch.load(
                os.path.join(options.modelOutFolder, options.pretrainedModelName + "_decoder" + "_%d.pth" % ep)))
            discriminator.load_state_dict(torch.load(
                os.path.join(options.modelOutFolder, options.pretrainedModelName + "_discriminator" + "_%d.pth" % ep)))
            z_discriminator.load_state_dict(torch.load(os.path.join(options.modelOutFolder,
                                                                    options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % ep)))
    if options.dataset == 'MNIST':
        dataloader = torch.utils.data.DataLoader(
            dset.MNIST(root='../../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
        val_dataloader = torch.utils.data.DataLoader(
            dset.MNIST('../../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
            batch_size=100, shuffle=False, num_workers=options.workers)
    if options.dataset == 'CIFAR10':
        dataloader = torch.utils.data.DataLoader(
            dset.CIFAR10(root='../../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))
                         ])),
            batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
        val_dataloader = torch.utils.data.DataLoader(
            dset.CIFAR10('../../data', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))
                         ])),
            batch_size=100, shuffle=False, num_workers=options.workers)

elif options.dataset == 'CelebA':
        if options.img_size == 0:
            celebA_imgsize = 64
        else:
            celebA_imgsize = options.img_size

        dataloader = torch.utils.data.DataLoader(
            custom_Dataloader(path=options.dataroot,
                              transform=transforms.Compose([
                                  transforms.CenterCrop(150),
                                  transforms.Scale((celebA_imgsize, celebA_imgsize)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ]), type='train'), batch_size=options.batchSize, shuffle=True,
            num_workers=options.workers)



    win_dict = utils.win_dict()
    line_win_dict = utils.win_dict()
    grad_line_win_dict = utils.win_dict()

    print("Training Start!")
    spend_time_log = []

    for epoch in range(options.epoch):
        for i, (data, _) in enumerate(dataloader, 0):
            real_cpu = data
            batch_size = real_cpu.size(0)
            input = Variable(real_cpu).cuda()
            disc_input = input.clone()

            real_label = Variable(torch.FloatTensor(batch_size).cuda())
            real_label.data.fill_(1)
            fake_label = Variable(torch.FloatTensor(batch_size).cuda())
            fake_label.data.fill_(0)
            noise_regularizer = Variable(torch.FloatTensor(real_cpu.shape)).cuda()
            noise_regularizer.data.fill_(1)
            time_calc.simple_time_start('%06d' % i)

            # autoencoder part
            if options.intergrationType != 'GANonly':
                if autoencoder_type == "VAE":
                    optimizerE.zero_grad()
                    optimizerD.zero_grad()
                    mu, logvar = encoder(input)
                    std = torch.exp(0.5 * logvar)
                    eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                    z = eps.mul(std).add_(mu)
                    x_recon = decoder(z)
                    err_recon, err_KLD = Variational_loss(x_recon, input.detach(), mu, logvar)
                    err = recon_weight * (err_recon + err_KLD)
                    err.backward(retain_graph=True)
                    generator_grad_AE = utils.torch_model_gradient(decoder.parameters())
                    optimizerE.step()
                    optimizerD.step()



            # adversarial training part  =======================================================================================
            # adversarial training, discriminator part
            if options.intergrationType != 'AEonly':  # GAN
                optimizerDiscriminator.zero_grad()

                    if autoencoder_type == 'VAE':
                        mu, logvar = encoder(generated_fake)
                        std = torch.exp(0.5 * logvar)
                        eps = Variable(torch.randn(std.size()), requires_grad=False).cuda()
                        z = eps.mul(std).add_(mu)
                    else:
                        z = encoder(generated_fake)
                    d_fake = discriminator(z)

                    d_real = discriminator(input)
                    noise = Variable(torch.FloatTensor(batch_size, nz)).cuda()
                    noise.data.normal_(0, 1)
                    generated_fake = decoder(noise.view(batch_size, nz, 1, 1))
                    d_fake = discriminator(generated_fake)

                balance_coef = torch.cat((d_fake, d_real), 0).mean()
                err_discriminator_real = BCE_loss(d_real.view(batch_size), real_label.view(batch_size))
                err_discriminator_fake = BCE_loss(d_fake.view(batch_size), fake_label.view(batch_size))
                err_discriminator_origin = err_discriminator_real + err_discriminator_fake
                err_discriminator = decoder_weight * err_discriminator_origin
                err_discriminator.backward(retain_graph=True)

                discriminator_grad = utils.torch_model_gradient(discriminator.parameters())
                optimizerDiscriminator.step()

                # adversarial training, generator part
                noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1)).cuda()
                noise.data.normal_(0, 1)
                    generated_fake = decoder(noise)
                    d_fake_2 = discriminator(generated_fake)
                else:
                    err_generator = BCE_loss(d_fake_2.view(batch_size), real_label.view(batch_size))
                    err_generator = decoder_weight * err_generator
                    optimizerD.zero_grad()
                    err_generator.backward(retain_graph=True)
                    generator_grad = utils.torch_model_gradient(decoder.parameters())
                    optimizerD.step()


                # visualize
                print('[%d/%d][%d/%d] d_real: %.4f d_fake: %.4f Balance : %.2f'
                      % (epoch, options.epoch, i, len(dataloader), d_real.data.mean(), d_fake_2.data.mean(),
                         balance_coef.data.mean()))
                # print(float(noise.data.view(noise.shape[0], -1).var(1).mean()))
                # print(float(noise.data.view(noise.shape[0], -1).mean(1).mean()))


            spend_time = time_calc.simple_time_end()
            spend_time_log.append(spend_time)

            line_win_dict = utils.draw_lines_to_windict(line_win_dict,
                                                                      [err_discriminator_real.data.mean(),
                                                                       err_discriminator_fake.data.mean(),
                                                                       err_generator.data.mean(),
                                                                       err_recon.data.mean(),
                                                                       err_KLD.data.mean(),
                                                                       0],
                                                                      ['D loss -real',
                                                                       'D loss -fake',
                                                                       'G loss',
                                                                       'recon loss',
                                                                       'KLD loss',
                                                                       'zero'], epoch, i, len(dataloader))

            grad_line_win_dict = utils.draw_lines_to_windict(grad_line_win_dict,
                                                                           [
                                                                               discriminator_grad,
                                                                               generator_grad,
                                                                               0],
                                                                           ['D gradient',
                                                                            'G gradient',
                                                                            'zero'],
                                                                           epoch, i, len(dataloader))

        line_win_dict = utils.draw_lines_to_windict(line_win_dict,
                                                                  [
                                                                      err_recon.data.mean(),
                                                                      err_KLD.data.mean(),
                                                                      0],
                                                                  [
                                                                      'recon loss',
                                                                      'KLD loss',
                                                                      'zero'], 0, epoch, 0)
        time_calc.mean_calc()

        # do checkpointing
        if epoch % options.save_tick == 0 and options.save:
            print("saving models")
            print(os.path.join(options.modelOutFolder,
                               options.pretrainedModelName + "_encoder" + "_%d.pth" % (epoch + ep)))
            torch.save(encoder.state_dict(), os.path.join(options.modelOutFolder,
                                                          options.pretrainedModelName + "_encoder" + "_%d.pth" % (
                                                                      epoch + ep)))
            torch.save(decoder.state_dict(), os.path.join(options.modelOutFolder,
                                                          options.pretrainedModelName + "_decoder" + "_%d.pth" % (
                                                                      epoch + ep)))
            torch.save(discriminator.state_dict(), os.path.join(options.modelOutFolder,
                                                                options.pretrainedModelName + "_discriminator" + "_%d.pth" % (
                                                                            epoch + ep)))
            torch.save(z_discriminator.state_dict(), os.path.join(options.modelOutFolder,
                                                                  options.pretrainedModelName + "_z_discriminator" + "_%d.pth" % (
                                                                              epoch + ep)))