import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import CharbonnierLoss
import cv2
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import models
from torchinfo import summary
# from torchsummary import summary

#--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
#--------------------------------------------


from PIL import Image
import os.path as osp
import torchvision.transforms.functional as TF

logger = logging.getLogger('base')


class SRModel(BaseModel):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()
        self.tt = 0
        self.cnt_t = 0

        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
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
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

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

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ

###########################################
        self.T_map_from_user_On = 0
        if self.T_map_from_user_On==1:
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]
            # T_map_name = osp.join('E:/util/cmap/cmap_' + img_name + '.png')
            # T_map_name = osp.join('E:/util/LPIPS_min_HR_DIV2K_valid_LR/' + img_name + '_TMap_best.png')
            # T_map_name = osp.join('E:/util/FxSR-PD-LPIPS_21BC/' + img_name + '_TMap_best.png')
            # T_map_name = osp.join('E:/util/FxSR-PD-LPIPS_21BC/' + img_name + '_TMap_best.png')
            # T_map_name = osp.join('E:/exp/FxSR-MG-LPIPS_test/FxSR-MG-LPIPS-VGG-LPIPSLoss-RRDB/results/FxSR-PD-OOE-RRDB-F_VGG22-L_BTMap-40000--_t100/DIV2K_val_Q100/' + img_name + '_cmap.png')
            # BSDS100 DIV2K_val_Q100
            T_map_name = osp.join(
                'E:/util/FxSR-PD-M1234-v2-DF2K_DIV2K_test-Flickr2K/General100/' + img_name + '_TMap_best_dn.png')
            # T_map_name = osp.join('E:/util/LPIPS_min_HR_DIV2K_valid_LR_Median_31/' + img_name + '_TMap_best.png')

            # image = Image.open(T_map_name)
            image = cv2.imread(T_map_name)
            x = TF.to_tensor(image[:,:,0])
            x.unsqueeze_(0)
            self.T_map_from_user = x
            self.T_map_from_user = torch.clamp(self.T_map_from_user, min=0.0, max=1.0)
###########################################

        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()

        self.texture_gain = torch.rand(1)
        gain_channel = torch.ones([16, 1, 32, 32])
        gain_channel = gain_channel * self.texture_gain

        self.var_L2 = torch.cat((self.var_L, gain_channel.to(self.device)), 1)

        self.fake_H = self.netG(self.var_L2)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, opt, logger, img_name):
        # pytorch_total_params = sum(p.numel() for p in self.netG.parameters())
        # print("pytorch_total_params = %f" % (pytorch_total_params))
        self.netG.eval()
        feat_act_info_print = 0
        test_map_gen = 0

        with torch.no_grad():
            # self.texture_gain = opt['network_G']['T_ctrl']
            self.t = opt.T_ctrl
            image_size = self.var_L.shape
            self.T_map = torch.ones([1, 1, image_size[2], image_size[3]]) * self.t

            if self.T_map_from_user_On == 1:
                self.T_map = self.T_map_from_user * self.t

            self.fake_H = self.netG((self.var_L, self.T_map.to(self.device)))
            # logger.info('{:20s}'.format(img_name))
        self.netG.train()


        #     # ---- parameter ---------
        #     # pytorch_total_params = sum(p.numel() for p in self.netG.parameters())
        #     # pytorch_total_params2 = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
        #     # print(pytorch_total_params)
        #     # print(pytorch_total_params2)
        #
        #     # self.var_L2 = torch.cat((self.var_L, gain_channel.to(self.device)), 1)
        #     # ---- profile ---------
        #     # Condi_input = torch.cat((self.var_L, gain_channel.cuda()), dim=1)
        #     # with torch.no_grad():
        #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
        #                  record_shapes=True) as prof:
        #         with record_function("model_inference"):
        #             # summary(self.net_g, (4, 128, 128))
        #
        #             # self.output = self.net_g((self.lq, gain_channel))
        #             # self.output = self.netG((self.var_L, gain_channel.cuda()))
        #             self.fake_H = self.netG((self.var_L, gain_channel.to(self.device)))
        #
        #             # batch_size = 1
        #             # summary(self.netG, input_size=(batch_size, 4, 128, 128))
        #
        #     # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        #     # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=1))
        #     # print(prof.key_averages().table(top_level_events_only=True))
        #     # print(prof.key_averages().total_average())
        #     # print(prof.key_averages().total_average().cuda_time_str)
        #     # print(prof.key_averages().total_average().self_cuda_time_total)
        #     self.cnt_t = self.cnt_t + 1
        #
        #     if self.cnt_t > 2:
        #         self.tt = self.tt + prof.key_averages().total_average().self_cuda_time_total
        #         self.tt_avg = self.tt / (self.cnt_t - 2)
        #         logger.info('%f %f %f %f' % (
        #         prof.key_averages().total_average().self_cuda_time_total, self.tt, self.cnt_t, self.tt_avg))
        #     # print(prof.key_averages().total_average().self_cuda_time_total_str)
        #     # print(prof.key_averages().total_average().cuda_time_total)
        #     mem_params = sum([param.nelement() * param.element_size() for param in self.netG.parameters()])
        #     mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.netG.buffers()])
        #     mem = mem_params + mem_bufs  # in bytes
        #     print('%f %f %f' %(mem_params, mem_bufs, mem))
        #     logger.info('{:20s}'.format(img_name))
        #
        # self.netG.train()

    def test_x8(self):
        # from https://github.com/thstkdgus35/EDSR-PyTorch
        self.netG.eval()

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [self.var_L]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])
        with torch.no_grad():
            sr_list = [self.netG(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        self.fake_H = output_cat.mean(dim=0, keepdim=True)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
