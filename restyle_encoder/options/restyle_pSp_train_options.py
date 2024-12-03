from restyle_encoder.options.train_options import TrainOptions
import perturbation.utils as utils

class RestylepSpTrainOptions(TrainOptions):

    def __init__(self):
        super(RestylepSpTrainOptions, self).__init__()

    def initialize(self):
        super(RestylepSpTrainOptions, self).initialize()
        # arguments for iterative encoding
        self.parser.add_argument('--n_iters_per_batch', default=5, type=int,help='Number of forward passes per batch during training')


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
