from abc import abstractmethod
import torchvision.transforms as transforms
import numpy as np


################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4}

class TransformsConfig(object):

	def __init__(self, config):
		self.config = config

	@abstractmethod
	def get_transforms(self):
		pass


class EncodeTransforms(TransformsConfig):

	def __init__(self, config):
		super(EncodeTransforms, self).__init__(config)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': None,
			'transform_test': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict
    
class EncodeTransformsAROME(TransformsConfig):
    def __init__(self, config, path, var_names=['u','v','t2m']):
        super(EncodeTransformsAROME, self).__init__(config)
        
        self.VI = [ var_dict[v] for v in var_names]
        
        Means = np.load(path+'Mean_4_var.npy')[self.VI]
        Maxs = np.load(path+'MaxNew_4_var.npy')[self.VI]
        
        self.means = list(tuple(Means))
        self.stds = list(tuple((1.0/0.95)*(Maxs)))
        
        
    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(self.means, self.stds)]),
            'transform_source': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.means, self.stds)]),
        'transform_test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)]),

        'transform_inference': transforms.Compose([transforms.ToTensor(),
                transforms.Normalize(self.means, self.stds)]),
        }
        return transforms_dict
    


class CarsEncodeTransforms(TransformsConfig):

	def __init__(self, config):
		super(CarsEncodeTransforms, self).__init__(config)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((192, 256)),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': None,
			'transform_test': transforms.Compose([
				transforms.Resize((192, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((192, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		}
		return transforms_dict
