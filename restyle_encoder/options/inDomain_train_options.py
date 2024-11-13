from options.train_options import TrainOptions
import perturbation.utils as utils

class inDomainTrainOptions(TrainOptions):

    def __init__(self):
        super(inDomainTrainOptions, self).__init__()

    def initialize(self):
        super(inDomainTrainOptions, self).initialize()

        self.parser.add_argument('--train_discriminator', action='store_true', help='Whether to train the discriminator')
        self.parser.add_argument('--adv_lambda', default=0.1, type=float,help='Adversarial Loss')
        
    
    def parse(self):
        config = self.parser.parse_args()
        return config

def createNamesFromLosses(config) :
    
    name ='loss_train'
    
    config_dict = vars(config)
    mspl = ''
    train_discriminator=''
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
            if value :
                mspl = '_multi_scale_PL'
        
        if 'train_discriminator' in arg :
            if value :
                train_discriminator='_train_discriminator'

    name = f'{name}_resnet={resnet}_network_type={network_type}{mspl}{train_discriminator}/'
    
    return name
