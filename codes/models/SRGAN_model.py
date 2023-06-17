import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss

logger = logging.getLogger('base')


class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)
            if opt['dist']:
                self.netD = DistributedDataParallel(self.netD,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD = DataParallel(self.netD)

            self.netG.train()
            self.netD.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'l2+F2':
                    self.cri_pix = (nn.MSELoss().to(self.device))*0.5 + (nn.MSELoss().to(self.device))*0.5
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                # self.netF_54 = networks.define_F_54(opt, use_bn=False).to(self.device)
                self.netF_44 = networks.define_F_44(opt, use_bn=False).to(self.device)
                self.netF_34 = networks.define_F_34(opt, use_bn=False).to(self.device)
                self.netF_22 = networks.define_F_22(opt, use_bn=False).to(self.device)
                self.netF_12 = networks.define_F_12(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    # self.netF_54 = DistributedDataParallel(self.netF_54,
                    #                                        device_ids=[torch.cuda.current_device()])
                    self.netF_44 = DistributedDataParallel(self.netF_44,
                                                           device_ids=[torch.cuda.current_device()])
                    self.netF_34 = DistributedDataParallel(self.netF_34,
                                                           device_ids=[torch.cuda.current_device()])
                    self.netF_22 = DistributedDataParallel(self.netF_22,
                                                           device_ids=[torch.cuda.current_device()])
                    self.netF_12 = DistributedDataParallel(self.netF_12,
                                                           device_ids=[torch.cuda.current_device()])
                else:
                    # self.netF_54 = DataParallel(self.netF_54)
                    self.netF_44 = DataParallel(self.netF_44)
                    self.netF_34 = DataParallel(self.netF_34)
                    self.netF_22 = DataParallel(self.netF_22)
                    self.netF_12 = DataParallel(self.netF_12)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        # texture_gain from probability distribution
        texture_gain = torch.rand(1)
        texture_gain = torch.clamp(texture_gain, min=0.0, max=1.0)

        self.t = texture_gain[0]
        LR_size = self.var_L.shape
        self.T_map = torch.ones([LR_size[0], 1, LR_size[2], LR_size[3]]) * self.t

        self.fake_H = self.netG((self.var_L, self.T_map))
        # ------------------------------------------------------------------------------------
        self.vgg_mix_w_12 = 0.0
        self.vgg_mix_w_22 = 0.0
        self.vgg_mix_w_34 = 0.0
        self.vgg_mix_w_44 = 0.0
        # self.vgg_mix_w_54=0.0

        self.adv_w = (self.t)
        self.l_pix_ww = 1.0 - (self.t)

        if (self.t <= 0.25):
            self.vgg_mix_w_12 = (self.t)
            self.vgg_mix_w_22 = 0
            self.vgg_mix_w_34 = 0
            self.vgg_mix_w_44 = 0

        elif (self.t <= 0.5):
            self.vgg_mix_w_12 = 0.25
            self.vgg_mix_w_22 = (self.t-0.25)
            self.vgg_mix_w_34 = 0
            self.vgg_mix_w_44 = 0

        elif (self.t <= 0.75):
            self.vgg_mix_w_12 = 0.25
            self.vgg_mix_w_22 = 0.25
            self.vgg_mix_w_34 = (self.t-0.5)
            self.vgg_mix_w_44 = 0

        else:
            self.vgg_mix_w_12 = 0.25
            self.vgg_mix_w_22 = 0.25
            self.vgg_mix_w_34 = 0.25
            self.vgg_mix_w_44 = (self.t - 0.75)
            # ------------------------------------------------------------------------------------

        l_g_total = 0
        l_g_pix = 0
        l_g_fea = 0
        l_g_fea_12 = 0
        l_g_fea_22 = 0
        l_g_fea_34 = 0
        l_g_fea_44 = 0
        l_g_gan = 0
        l_d_real = 0
        l_d_fake = 0

        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                cri_pix_ = self.cri_pix(self.fake_H, self.var_H)
                l_g_pix = cri_pix_ * (self.l_pix_w+self.l_pix_ww*20)
                l_g_total += l_g_pix

            if self.cri_fea:  # feature loss
                l_g_fea = 0

                real_fea_12 = self.netF_12(self.var_H).detach()
                fake_fea_12 = self.netF_12(self.fake_H)

                real_fea_22 = self.netF_22(self.var_H).detach()
                fake_fea_22 = self.netF_22(self.fake_H)

                real_fea_34 = self.netF_34(self.var_H).detach()
                fake_fea_34 = self.netF_34(self.fake_H)

                real_fea_44 = self.netF_44(self.var_H).detach()
                fake_fea_44 = self.netF_44(self.fake_H)

                l_g_fea_12 = self.cri_fea(fake_fea_12, real_fea_12) * (self.vgg_mix_w_12)
                l_g_fea_22 = self.cri_fea(fake_fea_22, real_fea_22) * (self.vgg_mix_w_22)
                l_g_fea_34 = self.cri_fea(fake_fea_34, real_fea_34) * (self.vgg_mix_w_34)
                l_g_fea_44 = self.cri_fea(fake_fea_44, real_fea_44) * (self.vgg_mix_w_44)
                l_g_fea = l_g_fea + l_g_fea_12 + l_g_fea_22 + l_g_fea_34 + l_g_fea_44

                l_g_total += l_g_fea


            pred_g_fake = self.netD(self.fake_H)
            if self.opt['train']['gan_type'] == 'gan':
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.var_ref).detach()
                l_g_gan = self.l_gan_w * (
                    self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                    self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_gan = l_g_gan * self.adv_w
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        if self.opt['train']['gan_type'] == 'gan':
            l_d_real = self.cri_gan(pred_d_real, True)
            l_d_fake = self.cri_gan(pred_d_fake, False)
            l_d_total = l_d_real + l_d_fake
        elif self.opt['train']['gan_type'] == 'ragan':
            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan'] = l_g_gan.item()

        self.log_dict['l_g_total'] = l_g_total
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_fea'] = l_g_fea.item()
        self.log_dict['l_g_gan'] = l_g_gan.item()
        self.log_dict['l_g_fea_12'] = l_g_fea_12.item()
        self.log_dict['l_g_fea_22'] = l_g_fea_22.item()
        self.log_dict['l_g_fea_34'] = l_g_fea_34.item()
        self.log_dict['l_g_fea_44'] = l_g_fea_44.item()

        self.log_dict['l_d_real'] = l_d_real.item()
        self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.t = 1.0
            LR_size = self.var_L.shape
            self.T_map = torch.ones([1, 1, LR_size[2], LR_size[3]]) * self.t
            self.fake_H = self.netG((self.var_L, self.T_map))
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)


