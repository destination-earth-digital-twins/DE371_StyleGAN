from restyle_encoder.configs import transforms_config
from restyle_encoder.configs.paths_config import dataset_paths


DATASETS = {
	
    'arome_encode' : {
       'transforms' : transforms_config.EncodeTransformsAROME,
       'train_source_root' : dataset_paths['arome_train'],
       'train_target_root' :  dataset_paths['arome_train'],
       'test_source_root' :  dataset_paths['arome_test'],
       'test_target_root' : dataset_paths['arome_test']
            }
}