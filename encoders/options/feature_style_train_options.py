from encoders.options.train_options import TrainOptions

class FeatureStyleTrainOptions(TrainOptions):

    def __init__(self):
        super(FeatureStyleTrainOptions, self).__init__()

    def initialize(self):
        super(FeatureStyleTrainOptions, self).initialize()
        self.parser.add_argument('--fake_image_on_batch', action='store_true',help='Whether to add fake image on batch for encoder')
        self.parser.add_argument('--l2_lambda_features', default=1, type=float,help='Loss on features')
        self.parser.add_argument('--reconstruction_loss_on_fake_sample', action='store_true',help='Whether to add a reconstruction loss on fake samples')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',help='Whether to add average latent vector to generate codes from encoder.')


    def parse(self):
        config = self.parser.parse_args()
        return config

def createNamesFromLosses(config) :
    
    name ='loss_train'
    
    config_dict = vars(config)
    mspl = ''
    start_from_latent_avg = ''
    encoder_type = ''
    fake_image_on_batch = ''
    reconstruction_loss_on_fake_sample = ''

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
        
        if 'fake_image_on_batch' in arg:
            fake_image_on_batch = value
        
        if 'reconstruction_loss_on_fake_sample' in arg:
            reconstruction_loss_on_fake_sample = value
            
    name = f'{name}_resnet={resnet}_network_type={network_type}{mspl}_start_from_latent_avg={start_from_latent_avg}_encoder_type={encoder_type}_fake_image_on_batch={fake_image_on_batch}_reconstruction_loss_on_fake_sample={reconstruction_loss_on_fake_sample}/'
    
    return name
