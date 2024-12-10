from encoders.options.train_options import TrainOptions
import perturbation.utils as utils

class HyperStyleTrainOptions(TrainOptions):

    def __init__(self):
        super(HyperStyleTrainOptions, self).__init__()

    def initialize(self):
        super(HyperStyleTrainOptions, self).initialize()
        # arguments for iterative encoding
        self.parser.add_argument('--n_iters_per_batch', default=1, type=int,
                                 help='Number of forward passes per batch during training')
        self.parser.add_argument('--training_on_fake_samples', action='store_true',
                                 help='Whether to train on fake samples')
        # self.parser.add_argument('--l2_lambda_on_fake_latent', default=1, type=float,help='L2 loss factor on fake features')
        # self.parser.add_argument('--perceptual_lambda_on_fake_samples', default=1, type=float,help='Perceptual Loss Factor on fake samples')
        self.parser.add_argument('--training_on_real_samples', action='store_true',
                                 help='Whether to train from real samples')
        
        self.parser.add_argument('--load_w_encoder', action='store_true', help='Whether to load the w e4e encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')
        self.parser.add_argument('--w_encoder_checkpoint_path', default='model_paths["e4e_w_encoder"]', type=str,
                                 help='Path to pre-trained W-encoder.')
        
        self.parser.add_argument('--layers_to_tune', default='0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24', type=str, 
                                 help='comma-separated list of which layers of the StyleGAN generator to tune')
    def parse(self):
        config = self.parser.parse_args()
        return config

def createNamesFromLosses(config) :
    
    name ='loss_train'
    
    config_dict = vars(config)
    training_on_real_samples = ''
    training_on_fake_samples = ''

    for arg, value in config_dict.items() :
        
        if 'lambda' in arg :
            if value !=0 :
                name = name + '_' + arg + '_' + str(value)
        
        if 'training_on_real_samples' in arg:
            training_on_real_samples = value
        
        if 'training_on_fake_samples' in arg:
            training_on_fake_samples = value

        if 'n_iters_per_batch' in arg :
            suffix = str(value)
    
    name = f'{name}_training_on_real={training_on_real_samples}_training_on_fake={training_on_fake_samples}_{suffix}_iter/'
    
    return name
