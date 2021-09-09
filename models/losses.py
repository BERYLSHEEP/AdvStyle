"""
-------------------------------------------------
   File Name:    Losses.py
   Author:       Zhonghao Huang
   Date:         2019/10/21
   Description:  Module implementing various loss functions
                 Copy from: https://github.com/akanimax/pro_gan_pytorch
-------------------------------------------------
"""

import numpy as np

import torch
import torch.nn as nn


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
        dis: Discriminator used for calculating the loss
             Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================

class StandardGAN(GANLoss):

    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, step_sizes):
        # small assertion:
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps)
        f_preds = self.dis(fake_samps)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds, dim =1),
            torch.ones(real_samps.shape[0]).to(device))

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds, dim =1),
            torch.zeros(fake_samps.shape[0]).to(device))

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, step_sizes):
        preds = self.dis(fake_samps)
        return self.criterion(torch.squeeze(preds, dim=1),
                              torch.ones(fake_samps.shape[0]).to(fake_samps.device))

class CrossEntropyGAN(GANLoss):
    """docstring for CrossEntropy"""
    def __init__(self, dis):
        super().__init__(dis)

        self.criterion = nn.CrossEntropyLoss()

    def dis_loss(self, samples, labels):
        _, preds = self.dis(samples)

        loss = self.criterion(preds, labels).to(samples.device)    
        return loss

        

class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, step_sizes):
        r_preds = self.dis(real_samps) + step_sizes
        f_preds = self.dis(fake_samps)

        loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, _, fake_samps, step_sizes):
        return -torch.mean(self.dis(fake_samps))


class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, step_sizes):
        # Obtain predictions
        r_preds = self.dis(real_samps) + 0.1*step_sizes
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))
        
        return loss

    def gen_loss(self, real_samps, fake_samps, step_sizes):
        # Obtain predictions
        r_preds = self.dis(real_samps) + 0.1*step_sizes
        f_preds = self.dis(fake_samps)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        return (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))

class IdentityGAN(GANLoss):
    """docstring for identityGAN"""
    def __init__(self, dis, lbd=0.1):
        super().__init__(dis)
        self.lbd =lbd

    def update_lbd(self, step):
        if step < 4000:
            self.lbd = 0.1
        elif step < 5000:
            self.lbd = 0.05
        elif step < 7000:
            self.lbd = 0.01
        elif step < 9000:
            self.lbd = 0.005
        else:
            self.lbd = 0.001

    def dis_loss(self, real_samps, fake_samps, labels, step):
        # cross entropy

        # Obtain predictions
        r_f, r_logit = self.dis(real_samps)
        f_f, f_logit = self.dis(fake_samps)

        cross_entropy = nn.CrossEntropyLoss()
        self.update_lbd(step)
        loss = cross_entropy(r_logit, labels) + self.lbd * cross_entropy(f_logit, labels)
        
        return loss

    def gen_loss(self, real_samps, fake_samps, loss_type='cos'):
        # cos feature

        # Obtain predictions
        r_f, r_logit = self.dis(real_samps)
        f_f, f_logit = self.dis(fake_samps)

        if loss_type == 'cos':
            loss = 1 - torch.cosine_similarity(r_f, f_f)
        elif loss_type == 'mse':
            MSE_Loss=nn.MSELoss(reduction="mean")
            loss = MSE_Loss(r_f, f_f)
        else: 
            raise ValueError('loss_type must be cos or mse, but {} is provide'.format(loss_type))
        return loss
        

class LogisticGAN(GANLoss):
    def __init__(self, dis):
        super().__init__(dis)

    # gradient penalty
    def R1Penalty(self, real_img):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(real_img)
        # real_logit = apply_loss_scaling(torch.sum(real_logit))
        real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_img,
                                         grad_outputs=torch.ones(real_logit.size()).to(real_img.device),
                                         create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
        # real_grads = undo_loss_scaling(real_grads)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def dis_loss(self, real_samps, fake_samps, step_sizes, r1_gamma=10.0):
        # Obtain predictions
        r_preds = self.dis(real_samps)+step_sizes
        f_preds = self.dis(fake_samps)

        loss = torch.mean(nn.Softplus()(f_preds)) + torch.mean(nn.Softplus()(-r_preds))

        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach()) * (r1_gamma * 0.5)
            loss += r1_penalty

        return loss

    def gen_loss(self, _, fake_samps, step_sizes):
        f_preds = self.dis(fake_samps)

        return torch.mean(nn.Softplus()(-f_preds))
