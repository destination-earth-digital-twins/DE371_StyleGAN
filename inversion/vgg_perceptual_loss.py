# -*- coding: utf-8 -*-
import torchvision
import torch
import torch.nn as nn

## adapted from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49#gistcomment-3347450

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, params=None, device=None):
        r''' Class that computes the Perceptual Loss based on VGG Network.

         As the original VGG has three channels there are different possibilities to forward the VGG Loss :
         - Solution 1 : Double FOR loop that iterates over batch and channel and replicates the input three times with 3-channel VGG forward
         - Solution 2 : Single FOR loop that iterates over channel and replicates the input three times with 3-channel VGG forward
         - Solution 3 : Single 3-channel VGG forward (Need for the input to have 3 channels)
         - Solution 4 : Single FOR loop that iterates over channel with an additional single channel CNN on top of the 3-channel VGG forward
         - Solution 5 : Single FOR loop that iterates over channel with a 1-channel VGG forward
         
         '''
        super(VGGPerceptualLoss, self).__init__()

        self.params=params
        self.device=device
        
        
        # Network initialization
        blocks = []
        self.flag_init_layer=True if params.vgg_computation=='sol4' else False
        self.flag_vgg_single_channel_input = True if params.vgg_computation=='sol5' else False

        if self.flag_init_layer and not self.flag_vgg_single_channel_input:
            # Solution 4
            self.grayscale_to_rgb = nn.Sequential(
                                    nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
                                    nn.ReLU()
                                    )
        vgg_network = torchvision.models.vgg16(weights=None)
        vgg_network.load_state_dict(torch.load(params.vgg_state_dict_path))

        if not self.flag_vgg_single_channel_input :
            # Solution 1, 2 and 3
            blocks.append(vgg_network.features[:4].eval())
        else :
            # Solution 4
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

        self.resize = self.params.resize_vgg_input

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Features memory in case they need to be memorized 
        # ex : For optimization procedure we have to compare the same input
        # features all along the optimization process. So there is no need
        # to compute the input features at each steps of optimization
        
        self.features_input_img=None
        self.styles_input_img=None 

    def forward_vgg_single_img(self, input_img, feature_layers=[0,1,2,3], style_layers=[], return_features=False):
        r''' Forward the VGG features and styles for a single image '''
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
    
    def compute_perceptual_features(self, img):
        r''' Compute the features of a single image with respect to the chosen solution and save them in the memory '''
        features = []
        styles = []
        if self.params.vgg_computation=='sol1':
            for i_mem in range(img.shape[0]):
                for i_var in range(img.shape[1]):
                    feature, style = self.forward_vgg_single_img(
                                                (img[i_mem, i_var, :, :]+1)/2,
                                                feature_layers = self.params.vgg_feature_layers,
                                                style_layers = self.params.vgg_style_layers
                    )
                    features.append(feature)
                    styles.append(style)

        elif self.params.vgg_computation in ['sol2', 'sol4', 'sol5']:
            for i_var in range(img.shape[1]):
                feature, style = self.forward_vgg_single_img(
                                                    (img[:, i_var, :, :]+1)/2,
                                                    feature_layers = self.params.vgg_feature_layers,
                                                    style_layers = self.params.vgg_style_layers
                        )
                features.append(feature)
                styles.append(style)

        elif self.params.vgg_computation == 'sol3':
            features, styles = self.forward_vgg_single_img(
                                                    (img+1)/2,
                                                    feature_layers = self.params.vgg_feature_layers,
                                                    style_layers = self.params.vgg_style_layers
                        )


        self.features_input_img=features
        self.styles_input_img=styles 

    
    def vgg_loss_given_features_and_target(self, target_img, feature_layers=[0,1,2,3], features_input_img=None, style_layers=[], styles_input_img=None, alpha_feature=1.0, alpha_style=0.01):
        r''' Computes the VGG Loss given features of an image and a target image '''

        features_target_img, styles_target_img = self.forward_vgg_single_img(target_img)

        loss = 0.0
        for i, _ in enumerate(self.blocks):
            if i in feature_layers:
                x = features_input_img[i]
                y = features_target_img[i]
                loss_features = torch.nn.functional.l1_loss(x, y)
                # print('loss_features', loss_features)
                loss += alpha_feature*loss_features
            if i in style_layers: # see https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
                gram_x = styles_input_img[i]
                gram_y = styles_target_img[i]
                loss_style = torch.nn.functional.l1_loss(gram_x, gram_y)
                # print('loss_style', loss_style)
                loss += alpha_style*loss_style
        return loss

    def vgg_loss_given_input_and_target(self, input_img, target_img, feature_layers=[0,1,2,3], style_layers=[], alpha_feature=1.0, alpha_style=0.01):
        r''' Computes the VGG Loss given an input image and a target image '''

        features_input_img, styles_input_img = self.forward_vgg_single_img(input_img)

        return self.vgg_loss_given_features_and_target(
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

        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            # TODO
            pass

        perceptual_loss = torch.tensor(0.).to(self.device)

        if input_img is not None:
            if self.params.vgg_computation=='sol1':
                for i_mem in range(img_gen.shape[0]):
                    for i_var in range(img_gen.shape[1]):
                        perceptual_loss += self.vgg_loss_given_input_and_target(
                            (img_gen[i_mem, i_var, :, :]+1)/2,
                            (input_img[i_mem, i_var, :, :]+1)/2,
                            feature_layers = self.params.vgg_feature_layers,
                            style_layers = self.params.vgg_style_layers,
                            alpha_feature = self.params.vgg_alpha_feature,
                            alpha_style = self.params.vgg_alpha_style
                        )

                perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]
            elif self.params.vgg_computation in ['sol2', 'sol4', 'sol5']:
                for i_var in range(img_gen.shape[1]):
                    perceptual_loss += self.vgg_loss_given_input_and_target( 
                        (img_gen[:, i_var, :, :]+1)/2,
                        (input_img[:, i_var, :, :]+1)/2,
                        feature_layers = self.params.vgg_feature_layers,
                        style_layers = self.params.vgg_style_layers,
                        alpha_feature = self.params.vgg_alpha_feature,
                        alpha_style = self.params.vgg_alpha_style
                    )
                perceptual_loss /= img_gen.shape[1]

            elif self.params.vgg_computation == 'sol3':
                perceptual_loss = self.vgg_loss_given_input_and_target(
                    (img_gen+1)/2,
                    (input_img+1)/2,
                    feature_layers = self.params.vgg_feature_layers,
                    style_layers = self.params.vgg_style_layers,
                    alpha_feature = self.params.vgg_alpha_feature,
                    alpha_style = self.params.vgg_alpha_style
                )
            else :
                raise NotImplementedError

        else :
            if self.features_input_img is None and self.styles_input_img is None :
                print('Warning: The features needs to be computed beforehand')
                raise ValueError
            
            if self.params.vgg_computation=='sol1':
                for i_mem in range(img_gen.shape[0]):
                    for i_var in range(img_gen.shape[1]):
                        features_input_img=self.features_input_img[i_mem+i_var]
                        if self.params.vgg_style_layers:
                            styles_input_img=self.styles_input_img[i_mem+i_var]
                        else :
                            styles_input_img=None
                        perceptual_loss += self.vgg_loss_given_features_and_target(
                            target_img=(img_gen[i_mem, i_var, :, :]+1)/2,
                            features_input_img=features_input_img, 
                            styles_input_img=styles_input_img,
                            feature_layers = self.params.vgg_feature_layers,
                            style_layers = self.params.vgg_style_layers,
                            alpha_feature = self.params.vgg_alpha_feature,
                            alpha_style = self.params.vgg_alpha_style
                        )
                       
                perceptual_loss /= img_gen.shape[0]*img_gen.shape[1]

            elif self.params.vgg_computation in ['sol2', 'sol4', 'sol5']:
                for i_var in range(img_gen.shape[1]):
                    features_input_img=self.features_input_img[i_var]
                    if self.params.vgg_style_layers:
                        styles_input_img=self.styles_input_img[i_var]
                    else :
                        styles_input_img=None
                    perceptual_loss += self.vgg_loss_given_features_and_target(
                        target_img=(img_gen[:, i_var, :, :]+1)/2,
                        features_input_img=features_input_img, 
                        styles_input_img=styles_input_img,
                        feature_layers = self.params.vgg_feature_layers,
                        style_layers = self.params.vgg_style_layers,
                        alpha_feature = self.params.vgg_alpha_feature,
                        alpha_style = self.params.vgg_alpha_style
                    )
                perceptual_loss /= img_gen.shape[1]

            elif self.params.vgg_computation == 'sol3':
                features_input_img=self.features_input_img
                if self.params.vgg_style_layers:
                    styles_input_img=self.styles_input_img
                else :
                    styles_input_img=None
                perceptual_loss += self.vgg_loss_given_features_and_target(
                    target_img=(img_gen+1)/2,
                    features_input_img=features_input_img, 
                    styles_input_img=styles_input_img,
                    feature_layers = self.params.vgg_feature_layers,
                    style_layers = self.params.vgg_style_layers,
                    alpha_feature = self.params.vgg_alpha_feature,
                    alpha_style = self.params.vgg_alpha_style
                )
            else :
                raise NotImplementedError
            
        return perceptual_loss