from inversion.perceptual_loss.perceptual_loss import PerceptualLoss, MultiPerceptualLoss
from argparse import ArgumentParser
import utils.utils as utils
import torch
parser = ArgumentParser()

parser.add_argument("--resize_input", type=float, default=0.0, help="resize input for vgg loss")
parser.add_argument("--network_type", type=str, default='vgg16')
parser.add_argument("--pre_trained", action='store_true')
parser.add_argument("--features_after_relu", action='store_true')
parser.add_argument("--channel_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
parser.add_argument("--network_dir", type=str, default='/project/home/p200177/DE_371/resources/network_for_perceptual_loss/', help="Insert a path")
parser.add_argument("--style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
parser.add_argument("--feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
parser.add_argument("--alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
parser.add_argument("--alpha_style", type=float, default=0.01, help="weight of the style loss")

config = parser.parse_args()
img1 = torch.rand((15,3,256,256)).to('cuda')
img2 = torch.rand((15,3,256,256)).to('cuda')
# for network_type in ['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16']:
#     for solution in ['sol1', 'sol2', 'sol3', 'sol4', 'sol5']:
#         print(f'testing {network_type} with {solution}')
#         config.network_type = network_type
#         config.channel_computation = solution
#         try :
#             perceptual_loss = PerceptualLoss(config=config, device='cuda')
#             loss = perceptual_loss.forward(img1,img2)
#             print(network_type, solution, loss)
#         except :
#             print(network_type, solution, 'failure')


config.network_type = ['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16']
perceptual_loss=MultiPerceptualLoss(config=config, device='cuda')
loss = perceptual_loss.forward(img1,img2)