"""
This file defines the core research contribution
"""
import math
import torch
from torch import nn

from gan.model.stylegan2 import Generator, Discriminator
from restyle_encoder.configs.paths_config import model_paths
from restyle_encoder.models.encoders import fpn_encoders, restyle_psp_encoders
from restyle_encoder.utils.model_utils import RESNET_MAPPING
from collections import OrderedDict

class inDomain(nn.Module):

    def __init__(self, config):
        super(inDomain, self).__init__()
        self.set_config(config)
        self.n_styles = int(math.log(self.config.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.config.output_size, 512, 8, channel_multiplier=2)
        self.discriminator = Discriminator(self.config.output_size, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((128, 128))

        # torch.save(self.encoder.state_dict(), '/project/scratch/p200177/DE_371/resources/pretrained_models/resnet34_random.pth')
        # raise NotImplementedError

        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.config.encoder_type == 'GradualStyleEncoder':
            encoder = fpn_encoders.GradualStyleEncoder(50, 'ir_se', self.n_styles, self.config)
        elif self.config.encoder_type == 'ResNetGradualStyleEncoder':
            encoder = fpn_encoders.ResNetGradualStyleEncoder(self.n_styles, self.config)
        elif self.config.encoder_type == 'BackboneEncoder':
            encoder = restyle_psp_encoders.BackboneEncoder(50, 'ir_se', self.n_styles, self.config)
        elif self.config.encoder_type == 'ResNetBackboneEncoder':
            encoder = restyle_psp_encoders.ResNetBackboneEncoder(self.n_styles, self.config)
        else:
            raise Exception(f'{self.config.encoder_type} is not a valid encoders')
        return encoder

    def load_weights(self):
        if self.config.checkpoint_path is not None:
            print(f'Loading ReStyle pSp from checkpoint: {self.config.checkpoint_path}')
            ckpt = torch.load(self.config.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self.__get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            encoder_ckpt = self.__get_encoder_checkpoint()
            if encoder_ckpt is not None:
                print(f'Loading encoder weights from resnet34: {self.config.stylegan_weights}')
                self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print(f'Loading generator weights from pretrained path: {self.config.stylegan_weights}')
            ckpt = torch.load(self.config.stylegan_weights)
            checkpoint = ckpt['g_ema']
            if 'module' in list(checkpoint.items())[0][0]: # juglling with Pytorch versioning and different module packaging
                ckpt_adapt = OrderedDict()
                for k in checkpoint.keys():
                    k0 = k[7:]
                    ckpt_adapt[k0] = checkpoint[k]
                self.decoder.load_state_dict(ckpt_adapt, strict=True)
            else:
                self.decoder.load_state_dict(checkpoint, strict=True)
            
            print(f'Loading discriminator weights from pretrained path: {self.config.stylegan_weights}')
            checkpoint = ckpt['d']
            if 'module' in list(checkpoint.items())[0][0]: # juglling with Pytorch versioning and different module packaging
                ckpt_adapt = OrderedDict()
                for k in checkpoint.keys():
                    k0 = k[7:]
                    ckpt_adapt[k0] = checkpoint[k]
                self.discriminator.load_state_dict(ckpt_adapt, strict=True)
            else:
                self.discriminator.load_state_dict(checkpoint, strict=True)

            self.__load_latent_avg(ckpt, repeat=self.n_styles)

    def forward(self, x, latent=None, resize=False, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None, average_code=False, input_is_full=False):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # residual step
            if x.shape[1] == 6 and latent is not None:
                # learn error with respect to previous iteration
                codes = codes + latent
            else :
                # first iteration is with respect to the avg latent code
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        if average_code:
            input_is_latent = True
        else:
            input_is_latent = (not input_code) or (input_is_full)

        images, result_latent, _ = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)
        if not input_code:
            if self.config.train_discriminator:
                discrim_out_real = self.discriminator(x)
                discrim_out_fake = self.discriminator(images)
            else :
                with torch.no_grad():
                    discrim_out_real = self.discriminator(x)
                    discrim_out_fake = self.discriminator(images)
        else :
            discrim_out_fake=None
            discrim_out_real=None

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent, discrim_out_real, discrim_out_fake
        else:
            return images, _, discrim_out_real, discrim_out_fake

    def set_config(self, config):
        self.config = config

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.config.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def __get_encoder_checkpoint(self):
        
        
        if "ffhq" in self.config.dataset_type:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # Transfer the RGB input of the irse50 network to the first 3 input channels of pSp's encoder
            if self.config.input_nc != 3:
                shape = encoder_ckpt['input_layer.0.weight'].shape
                altered_input_layer = torch.randn(shape[0], self.config.input_nc, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            return encoder_ckpt
        else:
            print('Loading encoders weights from resnet!')
            if self.config.random_resnet :
                return None
            else :
                encoder_ckpt = torch.load(model_paths['resnet34'])
                # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
                if self.config.input_nc != 3:
                    shape = encoder_ckpt['conv1.weight'].shape
                    altered_input_layer = torch.randn(shape[0], self.config.input_nc, shape[2], shape[3], dtype=torch.float32)
                    altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
                    encoder_ckpt['conv1.weight'] = altered_input_layer
                mapped_encoder_ckpt = dict(encoder_ckpt)
                for p, v in encoder_ckpt.items():
                    for original_name, psp_name in RESNET_MAPPING.items():
                        if original_name in p:
                            mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                            mapped_encoder_ckpt.pop(p)
                return encoder_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
