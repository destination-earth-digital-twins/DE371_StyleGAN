from encoders.options.train_options import TrainOptions

class inDomainTrainOptions(TrainOptions):

    def __init__(self):
        super(inDomainTrainOptions, self).__init__()

    def initialize(self):
        super(inDomainTrainOptions, self).initialize()

        self.parser.add_argument('--train_discriminator', action='store_true', help='Whether to train the discriminator')
        self.parser.add_argument('--adv_lambda', default=0.1, type=float,help='Adversarial Loss')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',help='Whether to add average latent vector to generate codes from encoder.')
    
    def parse(self):
        config = self.parser.parse_args()
        return config

def createNamesFromLosses(config) :
    
    name ='loss_train'
    
    config_dict = vars(config)
    mspl = ''
    train_discriminator=''
    start_from_latent_avg=''
    encoder_type = ''

    for arg, value in config_dict.items() :
        
        if 'lambda' in arg :
            if value !=0 :
                name = name + '_' + arg + '_' + str(value)
        
        if 'learning_rate' in arg :
            name = 'lr' + '_' + str(value)
        
        if 'network_type' in arg :
            network_type = value

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
        
        if 'start_from_latent_avg' in arg:
            start_from_latent_avg = value
        
        if 'encoder_type' in arg:
            encoder_type = value

    name = f'{name}_resnet={resnet}_network_type={network_type}{mspl}{train_discriminator}_start_from_latent_avg={start_from_latent_avg}_encoder_type={encoder_type}/'
    
    return name
