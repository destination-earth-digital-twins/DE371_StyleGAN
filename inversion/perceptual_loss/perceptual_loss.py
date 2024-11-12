# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from inversion.perceptual_loss.network_for_perceptual_loss import set_vgg16, set_vgg11, set_vgg13, set_vgg19, set_squeezenet1_1, set_vit_b_16
from inversion.perceptual_loss.network_for_perceptual_loss import set_alexnet, set_resnet101, set_resnet152, set_resnet18, set_resnet34, set_resnet50
# https://pytorch.org/vision/main/models.html

class MultiPerceptualLoss(torch.nn.Module):
    def __init__(self, config=None, device=None):
        super(MultiPerceptualLoss, self).__init__()
        assert isinstance(config.network_type,list)
        self.perceptual_losses = []
        network_type_list = deepcopy(config.network_type)
        for net_type in network_type_list :
            config.network_type = net_type
            self.perceptual_losses.append(deepcopy(PerceptualLoss(config=config, device=device)))

    def compute_perceptual_features(self, img):
        for id, perceptual_loss in enumerate(self.perceptual_losses) :
                perceptual_loss.compute_perceptual_features(img)

    def forward(self, img_gen, input_img=None, normalize=False):
        perceptual_loss_total = 0
        for id, perceptual_loss in enumerate(self.perceptual_losses) :
            loss =  perceptual_loss(img_gen, input_img, normalize)
            # print(perceptual_loss.config.network_type, loss)
            perceptual_loss_total += loss

        return perceptual_loss_total


class PerceptualLoss(torch.nn.Module):
    def __init__(self, config=None, device=None, multi_scale=False):
        r''' Class that computes the Perceptual Loss based on VGG Network.

         As the original VGG has three channels there are different possibilities to forward the VGG Loss :
         - Solution 1 : Double FOR loop that iterates over batch and channel and replicates the input three times with 3-channel VGG forward
         - Solution 2 : Single FOR loop that iterates over channel and replicates the input three times with 3-channel VGG forward
         - Solution 3 : Single 3-channel VGG forward (Need for the input to have 3 channels)
         - Solution 4 : Single FOR loop that iterates over channel with an additional single channel CNN on top of the 3-channel VGG forward
         - Solution 5 : Single FOR loop that iterates over channel with a 1-channel VGG forward
         
         '''
         # TODO : Add more description here
        super(PerceptualLoss, self).__init__()

        self.config=config
        self.device=device
        
        self.set_network()

        self.transform = torch.nn.functional.interpolate
        self.resize = self.config.resize_input

        # Features memory in case they need to be memorized 
        # ex : For optimization procedure we have to compare the same input
        # features all along the optimization process. So there is no need
        # to compute the input features at each steps of optimization
        
        self.features_input_img=None
        self.styles_input_img=None 

        self.multi_scale = multi_scale
        self.scaling_factor = [0]
        if multi_scale :
            for i in range(3):
                self.scaling_factor.append(2**i)
    
    def downscale(self, x, scale_times=1, mode='bilinear'):
        # print('before downscaling', x.shape)
        for _ in range(scale_times):
            x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode=mode)
        # print('after downscaling', x.shape)
        return x

    def set_network(self):
        print(f'Init Network {self.config.network_type}')
        if self.config.network_type == 'vgg16':
            blocks = set_vgg16(self)
        elif self.config.network_type == 'vgg11':
            blocks = set_vgg11(self)
        elif self.config.network_type == 'vgg19':
            blocks = set_vgg19(self)
        elif self.config.network_type == 'vgg13':
            blocks = set_vgg13(self)
        elif self.config.network_type == 'alexnet':
            blocks = set_alexnet(self)
        elif self.config.network_type == 'squeezenet1_1':
            blocks = set_squeezenet1_1(self)
        elif self.config.network_type == 'resnet18':
            blocks = set_resnet18(self)
        elif self.config.network_type == 'resnet34':
            blocks = set_resnet34(self)
        elif self.config.network_type == 'resnet50':
            blocks = set_resnet50(self)
        elif self.config.network_type == 'resnet101':
            blocks = set_resnet101(self)
        elif self.config.network_type == 'resnet152':
            blocks = set_resnet152(self)
        elif self.config.network_type == 'set_vit_b_16':
            blocks = set_vit_b_16(self)
        else :
            raise NotImplementedError

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))
        
        if self.config.channel_computation=='sol4':
            # Solution 4
            self.grayscale_to_rgb = nn.Sequential(
                                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
                                    nn.ReLU()
            )

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
    
    def forward_net_single_img(self, input_img, feature_layers=[0,1,2,3], style_layers=[], return_features=False):
        r''' Forward the Network features and styles for a single image '''

        if self.config.channel_computation=='sol4' and len(input_img.shape)==3: #sol4
            input_img = self.grayscale_to_rgb(input_img.unsqueeze(1))

        elif (input_img.shape[1] == 1 or len(input_img.shape)==2): #sol1
            # input size must be (B, 3, H, W). if input is (B, H, W) (grey scale image), we need to convert to (B, 3, H, W)
                input_img = input_img.repeat(1, 3, 1, 1)
        elif len(input_img.shape)==3 and self.config.channel_computation=='sol2': #sol2
            input_img = input_img.unsqueeze(1).repeat(1, 3, 1, 1)
        elif len(input_img.shape)==3 and self.config.channel_computation=='sol5': #sol5
            input_img = input_img.unsqueeze(1)
        
        if self.resize:
            input_img = self.transform(input_img, mode='bilinear', size=self.size_resize, align_corners=False)
            
        features = []
        styles= []

        # the samples have to be in range [0, 1] and normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        if self.config.channel_computation != 'sol5' :
            x = (input_img-self.mean) / self.std
        else :
            # grayscale imagenet's train dataset mean and standard deviation 
            
            grayscale_mean = 0.44531356896770125
            grayscale_std = 0.2692461874154524
            x = (input_img-grayscale_mean) / grayscale_std

        for i, block in enumerate(self.blocks):
            if self.config.network_type == 'set_vit_b_16' and i == 1:
                b, c, h, w = x.shape
                x = x.reshape(b,h*w,c)
            x = block(x)
            features.append(x)
            if i in style_layers: 
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                styles.append(gram_x)
        
        return features, styles
    
    def compute_perceptual_features(self, img):
        r''' Compute the features of a single image with respect to the chosen solution and save them in the memory '''
        features = []
        styles = []
        for scaling_factor in self.scaling_factor:
            if self.multi_scale:
                x = self.downscale(img, scaling_factor)
            else :
                x = img

            if self.config.channel_computation=='sol1':
                for i_mem in range(x.shape[0]):
                    for i_var in range(x.shape[1]):
                        feature, style = self.forward_net_single_img(
                                                    (x[i_mem, i_var, :, :]+1)/2,
                                                    feature_layers = self.config.feature_layers,
                                                    style_layers = self.config.style_layers
                        )
                        features.append(feature)
                        styles.append(style)

            elif self.config.channel_computation in ['sol2', 'sol4', 'sol5']:
                for i_var in range(x.shape[1]):
                    feature, style = self.forward_net_single_img(
                                                        (x[:, i_var, :, :]+1)/2,
                                                        feature_layers = self.config.feature_layers,
                                                        style_layers = self.config.style_layers
                            )
                    features.append(feature)
                    styles.append(style)

            elif self.config.channel_computation == 'sol3':
                features, styles = self.forward_net_single_img(
                                                        (x+1)/2,
                                                        feature_layers = self.config.feature_layers,
                                                        style_layers = self.config.style_layers
                            )


        self.features_input_img=features
        # print('self.features_input_img length :', len(self.features_input_img))
        self.styles_input_img=styles 

    
    def perceptual_loss_given_features_and_target(self, target_img, feature_layers=[0,1,2,3,4], features_input_img=None, style_layers=[], styles_input_img=None, alpha_feature=1.0, alpha_style=0.01):
        r''' Computes the Perceptual Loss given features of an image and a target image '''

        features_target_img, styles_target_img = self.forward_net_single_img(target_img)

        loss = 0.0
        for i, _ in enumerate(self.blocks):
            
            x = features_input_img[i]
            y = features_target_img[i]
            loss_features = torch.nn.functional.l1_loss(x, y)
            # print('loss_features', loss_features)
            loss += alpha_feature*loss_features

            if i in style_layers: 
                gram_x = styles_input_img[i]
                gram_y = styles_target_img[i]
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                # print('loss_style', loss_style)
                loss += alpha_style*loss_style
        return loss

    def perceptual_loss_given_input_and_target(self, input_img, target_img, feature_layers=[0,1,2,3], style_layers=[], alpha_feature=1.0, alpha_style=0.01):
        r''' Computes the VGG Loss given an input image and a target image '''

        features_input_img, styles_input_img = self.forward_net_single_img(input_img)

        return self.perceptual_loss_given_features_and_target(
            target_img=target_img,
            feature_layers=feature_layers,
            features_input_img=features_input_img,
            style_layers=style_layers,
            styles_input_img=styles_input_img,
            alpha_feature=alpha_feature,
            alpha_style=alpha_style
        )

    def forward(self, img_gen, input_img=None, normalize=False):
        r''' Computes the VGG loss between two images with respect to the chosen solution 
            NB: If input images are between -1 and +1, they will be normalized.
        '''

        if normalize: # turn on this flag if input is [-1, +1] so it can be adjusted to [01, 1]
            # TODO
            pass

        perceptual_loss = torch.tensor(0.).to(self.device)

        if input_img is not None:
            for scaling_factor in self.scaling_factor:
                if self.multi_scale:
                    x = self.downscale(img_gen, scaling_factor)
                    y = self.downscale(input_img, scaling_factor)
                else :
                    x = img_gen
                    y = input_img

                if self.config.channel_computation=='sol1':
                    for i_mem in range(x.shape[0]):
                        for i_var in range(x.shape[1]):
                            perceptual_loss += self.perceptual_loss_given_input_and_target(
                                (x[i_mem, i_var, :, :]+1)/2,
                                (y[i_mem, i_var, :, :]+1)/2,
                                feature_layers = self.config.feature_layers,
                                style_layers = self.config.style_layers,
                                alpha_feature = self.config.alpha_feature,
                                alpha_style = self.config.alpha_style
                            )

                    perceptual_loss /= x.shape[0]*x.shape[1]
                elif self.config.channel_computation in ['sol2', 'sol4', 'sol5']:
                    for i_var in range(x.shape[1]):
                        perceptual_loss += self.perceptual_loss_given_input_and_target( 
                            (x[:, i_var, :, :]+1)/2,
                            (y[:, i_var, :, :]+1)/2,
                            feature_layers = self.config.feature_layers,
                            style_layers = self.config.style_layers,
                            alpha_feature = self.config.alpha_feature,
                            alpha_style = self.config.alpha_style
                        )
                    perceptual_loss /= x.shape[1]

                elif self.config.channel_computation == 'sol3':
                    perceptual_loss = self.perceptual_loss_given_input_and_target(
                        (x+1)/2,
                        (y+1)/2,
                        feature_layers = self.config.feature_layers,
                        style_layers = self.config.style_layers,
                        alpha_feature = self.config.alpha_feature,
                        alpha_style = self.config.alpha_style
                    )
                else :
                    raise NotImplementedError

        else :
            if self.features_input_img is None and self.styles_input_img is None :
                print('Warning: The features needs to be computed beforehand')
                raise ValueError
            for id_scaling_factor, scaling_factor in enumerate(self.scaling_factor):
                if self.multi_scale:
                    x = self.downscale(img_gen, scaling_factor)
                else :
                    x = img_gen
                    id_scaling_factor = 0

                if self.config.channel_computation=='sol1':
                    for i_mem in range(x.shape[0]):
                        for i_var in range(x.shape[1]):
                            features_input_img=self.features_input_img[id_scaling_factor*(x.shape[1]+x.shape[0])+i_mem+i_var]
                            if self.config.style_layers:
                                styles_input_img=self.styles_input_img[id_scaling_factor*(x.shape[1]+x.shape[0])+i_mem+i_var]
                            else :
                                styles_input_img=None
                            perceptual_loss += self.perceptual_loss_given_features_and_target(
                                target_img=(x[i_mem, i_var, :, :]+1)/2,
                                features_input_img=features_input_img, 
                                styles_input_img=styles_input_img,
                                feature_layers = self.config.feature_layers,
                                style_layers = self.config.style_layers,
                                alpha_feature = self.config.alpha_feature,
                                alpha_style = self.config.alpha_style
                            )
                        
                    perceptual_loss /= x.shape[0]*x.shape[1]

                elif self.config.channel_computation in ['sol2', 'sol4', 'sol5']:
                    for i_var in range(x.shape[1]):
                        features_input_img=self.features_input_img[id_scaling_factor*x.shape[1]+i_var]
                        if self.config.style_layers:
                            styles_input_img=self.styles_input_img[id_scaling_factor*x.shape[1]+i_var]
                        else :
                            styles_input_img=None
                        perceptual_loss += self.perceptual_loss_given_features_and_target(
                            target_img=(x[:, i_var, :, :]+1)/2,
                            features_input_img=features_input_img, 
                            styles_input_img=styles_input_img,
                            feature_layers = self.config.feature_layers,
                            style_layers = self.config.style_layers,
                            alpha_feature = self.config.alpha_feature,
                            alpha_style = self.config.alpha_style
                        )
                    perceptual_loss /= x.shape[1]

                elif self.config.channel_computation == 'sol3':
                    features_input_img=self.features_input_img
                    if self.config.style_layers:
                        styles_input_img=self.styles_input_img
                    else :
                        styles_input_img=None
                    perceptual_loss += self.perceptual_loss_given_features_and_target(
                        target_img=(x+1)/2,
                        features_input_img=features_input_img, 
                        styles_input_img=styles_input_img,
                        feature_layers = self.config.feature_layers,
                        style_layers = self.config.style_layers,
                        alpha_feature = self.config.alpha_feature,
                        alpha_style = self.config.alpha_style
                    )
                else :
                    raise NotImplementedError
            
        return perceptual_loss