from collections import OrderedDict

import torch

Means, Stds = torch.Tensor([0.00323179, -0.00636883, -0.00447128]),\
                torch.Tensor([0.09430821,0.10637118, 0.1566861]) 

def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1', mode = 'net'):
    
    if net_type=='alex' : 
        # build url
        url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
            + f'master/lpips/weights/v{version}/{net_type}.pth'
    
        # download
        old_state_dict = torch.hub.load_state_dict_from_url(
            url, progress=True,
            map_location=None if torch.cuda.is_available() else torch.device('cpu')
        )
    
        # rename keys
        new_state_dict = OrderedDict()
        for key, val in old_state_dict.items():
            new_key = key
            new_key = new_key.replace('lin', '')
            new_key = new_key.replace('model.', '')
            new_state_dict[new_key] = val
    
    elif net_type=='discrim' :
        
        ckpt_path = '/scratch/mrmn/brochetc/GAN_2D/psp4arome_expe/285000.pt'
        
        old_state_dict = torch.load(ckpt_path, 
                                    map_location=None if torch.cuda.is_available() else torch.device('cpu'))['d']
        
        if mode=='net' :
            new_state_dict = OrderedDict(old_state_dict)
            for key, val in old_state_dict.items():
                if not 'convs' in key :
                    new_state_dict.pop(key)
                else :
                    new_state_dict.pop(key)
                    new_key = key
                    new_key = new_key.replace('convs.', 'layers.')
                    new_state_dict[new_key] = val
            new_state_dict['mean'] = Means[None, :, None, None]
            new_state_dict['std'] = Stds[None, :, None, None]
        
        elif mode == 'linear' :
            new_state_dict = OrderedDict(old_state_dict)
            for key, val in old_state_dict.items():
                if not 'convs' in key :
                    new_state_dict.pop(key)
                else :
                    new_state_dict.pop(key)
                    new_key = key
                    new_key = new_key.replace('convs.', 'layers.')
                    new_state_dict[new_key] = val
            new_state_dict['mean'] = Means[None, :, None, None]
            new_state_dict['std'] = Stds[None, :, None, None]
        
            
            
            
    return new_state_dict
