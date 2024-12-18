from argparse import ArgumentParser
import utils.utils as utils

class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # general setup
        self.parser.add_argument('--exp_dir', type=str, default='/project/scratch/p200177/DE_371/victorsanchez/results/encoder/test/', help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='arome_encode', type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument('--encoder_type', default='ResNetBackboneEncoder', type=str, help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=6, type=int, help='Number of input image channels to the ReStyle encoder. Should be set to 6.')
        self.parser.add_argument('--output_size', default=256, type=int, help='Output size of generator')
        self.parser.add_argument('--n_vars', default=3, type=int, help='Number of variables as channels')

        # batch size and dataloader works
        self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

        # optimizers
        self.parser.add_argument('--learning_rate', default=0.001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool,help='Whether to train the decoder model')
        self.parser.add_argument('--weight_decay', default = 0.0, type =float,help = 'Adding weight decay to the encoder')

        # loss lambdas
        self.parser.add_argument('--l2_lambda', default=1, type=float,help='L2 loss multiplier factor')
        self.parser.add_argument('--perceptual_lambda', default=1, type=float,help='L2 loss multiplier factor')
        self.parser.add_argument('--ffl_lambda', default=0, type=float,help='Focal Frequency Loss')

        # VGG parameters
        self.parser.add_argument("--multi_scale_perceptual_loss", action='store_true')
        self.parser.add_argument("--resize_input", type=float, default=0.0, help="resize input for vgg loss")
        self.parser.add_argument("--network_type", type=str, default='vgg16', choices=['vgg16','vgg11','vgg13','vgg19','alexnet','squeezenet1_1','resnet18','resnet34','resnet50','resnet101','resnet152','set_vit_b_16'])
        self.parser.add_argument("--pre_trained", action='store_true')
        self.parser.add_argument("--features_after_relu", action='store_true')
        self.parser.add_argument("--channel_computation", type=str, default='sol2', choices = ['sol1', 'sol2', 'sol3', 'sol4', 'sol5'], 
                        help="Either we compute layer by layer and member per member but we have to triple th einput to make it rgb or all in one (naive)")
        self.parser.add_argument("--network_dir", type=str, default='/project/home/p200177/DE_371/resources/network_for_perceptual_loss/', help="Insert a path")
        self.parser.add_argument("--style_layers", type=utils.str2intlist, default=[], help="style layers to include in vgg loss computation")
        self.parser.add_argument("--feature_layers", type=utils.str2intlist, default=[0,1,2,3], help="feature layers to include in vgg computation")
        self.parser.add_argument("--alpha_feature", type=float, default=1.0, help="weight of the feature/content loss")
        self.parser.add_argument("--alpha_style", type=float, default=0.01, help="weight of the style loss")
        
        # weights and checkpoint paths
        self.parser.add_argument('--stylegan_weights', default='/project/home/p200177/DE_371/resources/models/trained_generator/000024.pt', type=str,help='Path to StyleGAN model weights')
        self.parser.add_argument('--random_resnet', action='store_true')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to ReStyle model checkpoint')
        self.parser.add_argument('--resume_step', default=0, type=int,help='step number to resume from')

        # intervals for logging, validation, and saving
        self.parser.add_argument('--max_steps', default=10000, type=int,help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=50, type=int,help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=500, type=int,help='Validation interval')
        self.parser.add_argument('--save_interval', default=100, type=int,help='Model checkpoint interval')

        

    def parse(self):
        config = self.parser.parse_args()
        return config

def createNamesFromLosses(config) :
    
    name ='loss_train'
    
    config_dict = vars(config)
    mspl = ''

    for arg, value in config_dict.items() :
        
        if 'lambda' in arg :
            if value !=0 :
                name = name + '_' + arg + '_' + str(value)
        
        if 'learning_rate' in arg :
            name = 'lr' + '_' + str(value)
        
        if 'network_type' in arg :
            network_type = value

        if 'n_iters_per_batch' in arg :
            suffix = str(value)

        if 'random_resnet' in arg :
            if value :
                resnet = 'random'
            else :
                resnet = 'trained'
        
        if 'multi_scale_perceptual_loss' in arg :
            mspl = '_multi_scale_PL'
    name = f'{name}_resnet={resnet}_network_type={network_type}{mspl}_{suffix}_iter/'
    
    return name
