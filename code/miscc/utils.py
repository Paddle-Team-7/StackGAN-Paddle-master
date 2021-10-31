import os
import errno
import numpy as np

from copy import deepcopy
from miscc.config import cfg

from miscc.save import save_image
import paddle
import paddle.nn as nn


#############################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = 1 + logvar - mu.pow(2) - logvar.exp()
    # mu.pow(2).add(logvar.exp()).multiply(-1).add(1).add(logvar)
    KLD = paddle.mean(KLD_element) * (-0.5)
    return KLD


def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_imgs.shape[0]
    cond = conditions.detach()
    fake = fake_imgs.detach()
    # real_features = nn.parallel.data_parallel(netD, (real_imgs), gpus)
    # fake_features = nn.parallel.data_parallel(netD, (fake), gpus)
    real_features = netD(real_imgs)
    fake_features = netD(fake)
    # real pairs
    # inputs = (real_features, cond)
    # real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    real_logits = netD.get_cond_logits(real_features, cond)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    # inputs = (real_features[:(batch_size-1)], cond[1:])
    # wrong_logits = \
    #     nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    wrong_logits = netD.get_cond_logits(real_features[:(batch_size-1)], cond[1:])
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    # inputs = (fake_features, cond)
    # fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    fake_logits = netD.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        # real_logits = \
        #     nn.parallel.data_parallel(netD.get_uncond_logits,
        #                               (real_features), gpus)
        real_logits = netD.get_uncond_logits(real_features)
        # fake_logits = \
        #     nn.parallel.data_parallel(netD.get_uncond_logits,
        #                               (fake_features), gpus)
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.detach()[0], errD_wrong.detach()[0], errD_fake.detach()[0]


def compute_generator_loss(netD, fake_imgs, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    # fake_features = nn.parallel.data_parallel(netD, (fake_imgs), gpus)
    fake_features = netD(fake_imgs)
    # fake pairs
    # inputs = (fake_features, cond)
    # fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    fake_logits = netD.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        # fake_logits = \
        #     nn.parallel.data_parallel(netD.get_uncond_logits,
        #                               (fake_features), gpus)
        netD.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        m.weight.set_value(paddle.normal(0.0, 0.02, m.weight.shape))
    elif classname.find('BatchNorm') != -1:
        # m.weight.data.normal_(1.0, 0.02)
        # m.bias.data.fill_(0)
        m.weight.set_value(paddle.normal(1.0, 0.02, m.weight.shape))
        m.bias.set_value(paddle.zeros_like(m.bias))
    elif classname.find('Linear') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        m.weight.set_value(paddle.normal(0.0, 0.02, m.weight.shape))
        if m.bias is not None:
            # m.bias.data.fill_(0.0)
            m.bias.set_value(paddle.zeros_like(m.bias))


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        save_image(
            fake.detach(), '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        save_image(
            fake.detach(), '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def save_model(netG, netD, epoch, model_dir):
    paddle.save(
        netG.state_dict(),
        '%s/netG_epoch_%d.pdparams' % (model_dir, epoch))
    paddle.save(
        netD.state_dict(),
        '%s/netD_epoch_last.pdparams' % (model_dir))
    print('Save G/D models')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
