# -*- coding: utf-8 -*-
import torchvision
import torch

## adapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#gistcomment-3347450
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False, pre_trained=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT' if pre_trained else None).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT'if pre_trained else None).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT'if pre_trained else None).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT'if pre_trained else None).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_img, target_img, feature_layers=[0,1,2,3], style_layers=[], alpha_feature=1.0, alpha_style=0.01):
        if input_img.shape[1] != 3: # input size must be (B, 3, H, W). if input is (B, H, W) (grey scale image), we need to convert to (B, 3, H, W)
            input_img = input_img.repeat(1, 3, 1, 1)
            target_img = target_img.repeat(1, 3, 1, 1)

        if self.resize:
            input_img = self.transform(input_img, mode='bilinear', size=(224, 224), align_corners=False)
            target_img = self.transform(target_img, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0

        # the samples have to be in range [0, 1] and normalized using
        # mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
        x = (input_img-self.mean) / self.std
        y = (target_img-self.mean) / self.std

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += alpha_feature*torch.nn.functional.l1_loss(x, y)
            if i in style_layers: # see https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += alpha_style*torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss