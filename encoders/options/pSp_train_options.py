from encoders.options.train_options import TrainOptions

class pSpTrainOptions(TrainOptions):

    def __init__(self):
        super(pSpTrainOptions, self).__init__()

    def initialize(self):
        super(pSpTrainOptions, self).initialize()
        self.parser.add_argument('--start_from_latent_avg', action='store_true',help='Whether to add average latent vector to generate codes from encoder.')
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
    mspl = ''
    start_from_latent_avg = ''
    encoder_type = ''
    training_on_real_samples = ''

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
            mspl = '_multi_scale_PL'
        
        if 'start_from_latent_avg' in arg:
            start_from_latent_avg = value
        
        if 'encoder_type' in arg:
            encoder_type = value
        
        if 'training_on_real_samples' in arg:
            training_on_real_samples = value
            
    name = f'{name}_resnet={resnet}_network_type={network_type}{mspl}_start_from_latent_avg={start_from_latent_avg}_encoder_type={encoder_type}_training_on_real_samples={training_on_real_samples}/'
    
    return name
