# -*- coding: utf-8 -*-
import torchvision
import torch
import torch.nn as nn


## adapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#gistcomment-3347450
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False, state_dict_path='', init_layer=False, vgg_single_channel_input=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        self.flag_init_layer=init_layer
        self.flag_vgg_single_channel_input = vgg_single_channel_input
        if init_layer and not vgg_single_channel_input:
            self.grayscale_to_rgb = nn.Sequential(
                                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
                                    nn.ReLU()
                                    )
        vgg_network = torchvision.models.vgg16(weights=None)
        vgg_network.load_state_dict(torch.load(state_dict_path))

        if not vgg_single_channel_input :
            blocks.append(vgg_network.features[:4].eval())
        else :
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)).eval(),
                    vgg_network.features[1].eval(),
                    vgg_network.features[2].eval(),
                    vgg_network.features[3].eval()
                )
            )
        blocks.append(vgg_network.features[4:9].eval())
        blocks.append(vgg_network.features[9:16].eval())
        blocks.append(vgg_network.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward_single_img(self, input_img, feature_layers=[0,1,2,3], style_layers=[]):
        
        if self.flag_init_layer and len(input_img.shape)==3: #sol4
            input_img = self.grayscale_to_rgb(input_img.unsqueeze(1))

        elif (input_img.shape[1] == 1 or len(input_img.shape)==2): #sol1
            # input size must be (B, 3, H, W). if input is (B, H, W) (grey scale image), we need to convert to (B, 3, H, W)
                input_img = input_img.repeat(1, 3, 1, 1)
        elif len(input_img.shape)==3 and not self.flag_vgg_single_channel_input: #sol2
            input_img = input_img.unsqueeze(1).repeat(1, 3, 1, 1)
        elif len(input_img.shape)==3 and self.flag_vgg_single_channel_input: #sol5
            input_img = input_img.unsqueeze(1)
        if self.resize:
            input_img = self.transform(input_img, mode='bilinear', size=(224, 224), align_corners=False)
            
        features = []
        styles= []

        # the samples have to be in range [0, 1] and normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        if not self.flag_vgg_single_channel_input:
            x = (input_img-self.mean) / self.std
        else :
            # grayscale imagenet's train dataset mean and standard deviation 
            # from https://stackoverflow.com/questions/65699020/calculate-standard-deviation-for-grayscale-imagenet-pixel-values-with-rotation-m/65717887#65717887
            grayscale_mean = 0.44531356896770125
            grayscale_std = 0.2692461874154524
            x = (input_img-grayscale_mean) / grayscale_std

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in feature_layers:
                features.append(x)
            if i in style_layers: # see https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                styles.append(gram_x)
        return features, styles    

    def forward_given_features(self, target_img, feature_layers=[0,1,2,3], features_input_img=None, style_layers=[], styles_input_img=None, alpha_feature=1.0, alpha_style=0.01):
        
        if self.flag_init_layer and len(target_img.shape)==3: #sol4
            target_img = self.grayscale_to_rgb(target_img.unsqueeze(1))

        elif (target_img.shape[1] == 1 or len(target_img.shape)==2): #sol1
            # input size must be (B, 3, H, W). if input is (B, H, W) (grey scale image), we need to convert to (B, 3, H, W)
                target_img = target_img.repeat(1, 3, 1, 1)
        elif len(target_img.shape)==3 and not self.flag_vgg_single_channel_input: #sol2
            target_img = target_img.unsqueeze(1).repeat(1, 3, 1, 1)
        elif len(target_img.shape)==3 and self.flag_vgg_single_channel_input: #sol5
            target_img = target_img.unsqueeze(1)
        if self.resize:
            target_img = self.transform(target_img, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0

        # the samples have to be in range [0, 1] and normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        if not self.flag_vgg_single_channel_input:
            y = (target_img-self.mean) / self.std
        else :
            # grayscale imagenet's train dataset mean and standard deviation 
            # from https://stackoverflow.com/questions/65699020/calculate-standard-deviation-for-grayscale-imagenet-pixel-values-with-rotation-m/65717887#65717887
            grayscale_mean = 0.44531356896770125
            grayscale_std = 0.2692461874154524
            y = (target_img-grayscale_mean) / grayscale_std

        for i, block in enumerate(self.blocks):
            x = features_input_img[i]
            y = block(y)
            if i in feature_layers:
                loss_features = torch.nn.functional.l1_loss(x, y)
                # print('loss_features', loss_features)
                loss += alpha_feature*loss_features
            if i in style_layers: # see https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = styles_input_img[i]
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                # print('loss_style', loss_style)
                loss += alpha_style*loss_style
        return loss

    def forward(self, input_img, target_img, feature_layers=[0,1,2,3], style_layers=[], alpha_feature=1.0, alpha_style=0.01):
        
        if self.flag_init_layer and len(input_img.shape)==3: #sol4
            input_img = self.grayscale_to_rgb(input_img.unsqueeze(1))
            target_img = self.grayscale_to_rgb(target_img.unsqueeze(1))

        elif (input_img.shape[1] == 1 or len(input_img.shape)==2): #sol1
            # input size must be (B, 3, H, W). if input is (B, H, W) (grey scale image), we need to convert to (B, 3, H, W)
                input_img = input_img.repeat(1, 3, 1, 1)
                target_img = target_img.repeat(1, 3, 1, 1)
        elif len(input_img.shape)==3 and not self.flag_vgg_single_channel_input: #sol2
            input_img = input_img.unsqueeze(1).repeat(1, 3, 1, 1)
            target_img = target_img.unsqueeze(1).repeat(1, 3, 1, 1)
        elif len(input_img.shape)==3 and self.flag_vgg_single_channel_input: #sol5
            input_img = input_img.unsqueeze(1)
            target_img = target_img.unsqueeze(1)
        if self.resize:
            input_img = self.transform(input_img, mode='bilinear', size=(224, 224), align_corners=False)
            target_img = self.transform(target_img, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0

        # the samples have to be in range [0, 1] and normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        if not self.flag_vgg_single_channel_input:
            x = (input_img-self.mean) / self.std
            y = (target_img-self.mean) / self.std
        else :
            # grayscale imagenet's train dataset mean and standard deviation 
            # from https://stackoverflow.com/questions/65699020/calculate-standard-deviation-for-grayscale-imagenet-pixel-values-with-rotation-m/65717887#65717887
            grayscale_mean = 0.44531356896770125
            grayscale_std = 0.2692461874154524
            x = (input_img-grayscale_mean) / grayscale_std
            y = (target_img-grayscale_mean) / grayscale_std

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss_features = torch.nn.functional.l1_loss(x, y)
                # print('loss_features', loss_features)
                loss += alpha_feature*loss_features
            if i in style_layers: # see https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                # print('loss_style', loss_style)
                loss += alpha_style*loss_style
        return loss