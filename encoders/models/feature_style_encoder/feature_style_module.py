"""
This file defines the core research contribution
"""
import math
import torch
from torch import nn

from gan.model.stylegan2 import Generator
from encoders.configs.paths_config import model_paths
from encoders.models.feature_style_encoder.feature_style_encoder import fs_encoder
from collections import OrderedDict

class FeatureStyleModule(nn.Module):

    def __init__(self, config):
        super(FeatureStyleModule, self).__init__()
        self.set_config(config)
        self.n_styles = int(math.log(self.config.output_size, 2)) * 2 - 2
        # Define architecture
        self.idx_k = 6 # See paper https://arxiv.org/pdf/2202.02183 Instance 1 - 7 / Instance 2 - 6
        self.encoder = fs_encoder(n_styles=self.n_styles)
        self.decoder = Generator(self.config.output_size, 512, 8, channel_multiplier=2)

        # Load weights if needed
        self.load_weights()


    def load_weights(self):
        if self.config.checkpoint_path is not None:
            print(f'Loading Encoder and Generator from checkpoint: {self.config.checkpoint_path}')
            ckpt = torch.load(self.config.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self.__get_keys(ckpt, 'encoder'), strict=False)
            self.decoder.load_state_dict(self.__get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            
            print(f'Loading decoder weights from pretrained path: {self.config.stylegan_weights}')
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
            self.__load_latent_avg(ckpt, repeat=self.n_styles)

    def forward(self,
                x,
                feature_scale=0.0001,
                train=True,
                return_latent=False
    ):
        if self.config.fake_image_on_batch :
            # generate synthetic images
            z = torch.randn((x.shape[0], 512), device=self.config.device).detach()
            synthetic_img, synthetic_w, _ = self.decoder([z], return_latents=True, randomize_noise=False)
            synthetic_img = synthetic_img.detach()
            
            # Concat synthetic and real data
            img = torch.cat([synthetic_img, x], dim=0).detach()
        else :
            img = x
      
        # Reconstruction
        w_recon, fea = self.encoder(img)
        if self.config.start_from_latent_avg:
            w_recon = w_recon + self.latent_avg.repeat(w_recon.shape[0], 1, 1)

        if train: 
            features = None
        else :
            features = [None]*self.idx_k + [fea] + [None]*(13-self.idx_k) 
        
        # Recontruction from generate image
        # we pass the ground truth noises into the generator, so that the encoder can focus on the information encoded by the latent code
        x_recon, fea_recon, _ = self.decoder(
                                            [w_recon],
                                            randomize_noise=False,
                                            input_is_latent=True,
                                            return_features=True,
                                            features_in=features,
                                            feature_scale=feature_scale
        )

        fea_recon = fea_recon[self.idx_k].detach() 

        # Recontruction from real image - we pass random noises into the generator
        x_recon_2, _, _ = self.decoder(
                                    [w_recon],
                                    randomize_noise=True,
                                    input_is_latent=True,
                                    features_in=features,
                                    feature_scale=feature_scale
        )

        
        if return_latent :
            return img, fea, fea_recon, x_recon, x_recon_2, w_recon
        else :
            return img, fea, fea_recon, x_recon, x_recon_2

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
            print('Loading encoders weights from resnet50!')
            if self.config.random_resnet :
                return None
            else :
                encoder_ckpt = torch.load(model_paths['resnet50'])
                # Transfer the RGB input of the resnet34 network to the first 3 input channels of pSp's encoder
                # if self.config.input_nc != 3:
                #     shape = encoder_ckpt['conv1.weight'].shape
                #     altered_input_layer = torch.randn(shape[0], self.config.input_nc, shape[2], shape[3], dtype=torch.float32)
                #     altered_input_layer[:, :3, :, :] = encoder_ckpt['conv1.weight']
                #     encoder_ckpt['conv1.weight'] = altered_input_layer
                # mapped_encoder_ckpt = dict(encoder_ckpt)
                # for p, v in encoder_ckpt.items():
                #     for original_name, psp_name in RESNET_MAPPING.items():
                #         if original_name in p:
                #             mapped_encoder_ckpt[p.replace(original_name, psp_name)] = v
                #             mapped_encoder_ckpt.pop(p)
                return encoder_ckpt

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
