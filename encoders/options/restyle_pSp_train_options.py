from encoders.options.train_options import TrainOptions
import utils.utils as utils

class RestylepSpTrainOptions(TrainOptions):

    def __init__(self):
        super(RestylepSpTrainOptions, self).__init__()

    def initialize(self):
        super(RestylepSpTrainOptions, self).initialize()
        # arguments for iterative encoding
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,help='Number of forward passes per batch during training')
        self.parser.add_argument('--training_on_fake_samples', action='store_true',help='Whether to train on fake samples')
        self.parser.add_argument('--l2_lambda_on_fake_latent', default=1, type=float,help='L2 loss factor on fake features')
        self.parser.add_argument('--perceptual_lambda_on_fake_samples', default=1, type=float,help='Perceptual Loss Factor on fake samples')
        self.parser.add_argument('--training_on_real_samples', action='store_true',help='Whether to train from real samples')

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
