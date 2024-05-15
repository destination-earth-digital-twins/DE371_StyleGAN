# StyleGAN4AROME
A new proposal to generate ensemble predictions matching the AROME-EPS dataset. Even though several models are available for training, the current research is focusing on stylegan2 network (see the [original implementation](https://github.com/NVlabs/stylegan2) and the [pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch). The goal is to enrich the AROME-EPS dataset by generating samples mimicking the training data (i.e. to re-sample data from the latent distribution).  
A non-exhaustive diagram representing the global architecture is available on [Google Drive](https://drive.google.com/file/d/12Yidj0SBGblODHQIHi9Gf1WzNTqLoiJq/view?usp=sharing).  
Most of the core code is taken as is from [Rosinality's stylegan2-pytorch github page](https://github.com/rosinality/stylegan2-pytorch) and adapted to  run on Meteo France clusters. 
See the paper here: ??
Authors: C. Brochet, G. Moldovan

# Repository Structure

| Path | Description |
| --- | --- |
|[styleganpnria](https://github.com/flyIchtus/styleganPNRIA)|Root folder of the repository|
|&ensp;&ensp;&boxvr;&nbsp; [Dockerfile](https://github.com/flyIchtus/styleganpnria/blob/main/Dockerfile)|Docker configuration file|
|&ensp;&ensp;&boxvr;&nbsp; [docs](https://github.com/your-username/styleganpnria/blob/main/docs)|Documentation folder|
|&ensp;&ensp;&boxvr;&nbsp; [expe_init.py](https://github.com/your-username/styleganpnria/blob/main/expe_init.py)|Script for launching experiments.|
|&ensp;&ensp;&boxvr;&nbsp; [gan](https://github.com/flyIchtus/styleganpnria/blob/main)|Main folder for StyleGAN code|
|&ensp;&ensp;&boxvr;&nbsp; [gan_2_ae](https://github.com/flyIchtus/styleganpnria/blob/main/gan_2_ae)|Subfolder with gan_2_ae-related files|
|&ensp;&ensp;&boxvr;&nbsp; [grandensemble_inversion.py](https://github.com/your-username/styleganpnria/blob/main/grandensemble_inversion.py)|Script for grand ensemble inversion|
|&ensp;&ensp;&boxvr;&nbsp; [grandensemble_perturbation.py](https://github.com/flyIchtus/styleganpnria/blob/main/grandensemble_perturbation.py)|Script for grand ensemble perturbation|
|&ensp;&ensp;&boxvr;&nbsp; [hyperparams](https://github.com/flyIchtus/styleganpnria/blob/main/hyperparams)|Subfolder with hyperparameter files|
|&ensp;&ensp;&boxvr;&nbsp; [__init__.py](https://github.com/flyIchtus/styleganpnria/blob/main/__init__.py)|Python package initialization file|
|&ensp;&ensp;&boxvr;&nbsp; [main_gan.py](https://github.com/flyIchtus/styleganpnria/blob/main/main_gan.py)|Script for main GAN operations|
|&ensp;&ensp;&boxvr;&nbsp; [main_inversion.py](https://github.com/flyIchtus/styleganpnria/blob/main/main_inversion.py)|Script for inversion operations|
|&ensp;&ensp;&boxvr;&nbsp; [main_perturbation.py](https://github.com/flyIchtus/styleganpnria/blob/main/main_perturbation.py)|Script for generation of GAN ensembles|
|&ensp;&ensp;&boxvr;&nbsp; [metrics4arome](https://github.com/flyIchtus/styleganpnria/blob/main/metrics4arome)|Metrics for on-the-fly evaluation|
|&ensp;&ensp;&boxvr;&nbsp; [metrics4ensemble](https://github.com/flyIchtus/styleganpnria/blob/main/metrics4ensemble)|Metrics for ensemble evaluation|
|&ensp;&ensp;&boxvr;&nbsp; [metric_tests_exec.py](https://github.com/flyIchtus/styleganpnria/blob/main/metric_tests_exec.py)|Script for metric tests (execution)|
|&ensp;&ensp;&boxvr;&nbsp; [metric_tests_scat.py](https://github.com/flyIchtus/styleganpnria/blob/main/metric_tests_scat.py)|Script for metric tests (scatter)|
|&ensp;&ensp;&boxvr;&nbsp; [optuna_trial.py](https://github.com/flyIchtus/styleganpnria/blob/main/optuna_trial.py)|Script for Optuna trials|
|&ensp;&ensp;&boxvr;&nbsp; [perturbation](https://github.com/flyIchtus/styleganpnria/blob/main/perturbation)|Subfolder with perturbation-related files|
|&ensp;&ensp;&boxvr;&nbsp; [preprocessing](https://github.com/flyIchtus/styleganpnria/blob/main/preprocessing)|Subfolder with preprocessing-related files|
|&ensp;&ensp;&boxvr;&nbsp; [README.md](https://github.com/flyIchtus/styleganpnria/blob/main/README.md)|Main repository documentation|
|&ensp;&ensp;&boxur;&nbsp; [requirements.txt](https://github.com/flyIchtus/styleganpnria/blob/main/requirements.txt)|List of required Python packages|




## The AROME-EPS Dataset

The dataset comprises 516 AROME ensemble forecasts covering the period from June 15th, 2020, to November 12th, 2021. Each ensemble forecast is composed of 16 members and includes lead times at 1-hour intervals, ranging up to 45 hours. It follows that [516x45x16=371520]() individual samples are available for training if each members of the enseble at a given lead time is considered individually.

The data is restricted to a region encompassing the south and center of France with a resolution of [256x256]. Three variables are here considered: the horizontal (u) and vertical (v) components of the wind speed vector at 10 meters and the temperature at 2 meters (t2m). Each individual sample can be conceptualized as a tensor with 3 channels, a width of 256 and a height of 256 [3, 256, 256].

To efficiently load and organize the dataset, a metadata CSV file is utilized. The file structure is illustrated below:

| Name          | Importance | PosX | PosY | Date       | LeadTime | Member |
|---------------|------------|------|------|------------|----------|--------|
| ...           | ...        | ...  | ...  | ...        | ...      | ...    |
| _sample1440   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 0      |
| _sample1441   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 1      |
| _sample1442   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 2      |
| _sample1443   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 3      |
| _sample1444   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 4      |
| _sample1445   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 5      |
| _sample1446   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 6      |
| _sample1447   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 7      |
| _sample1448   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 8      |
| _sample1449   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 9      |
| _sample1450   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 10     |
| _sample1451   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 11     |
| _sample1452   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 12     |
| _sample1453   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 13     |
| _sample1454   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 14     |
| _sample1455   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 15     |
| _sample1456   | 1,0        | 256  | 256  | 2021-06-02 | 1        | 0      |
| _sample1457   | 1,0        | 256  | 256  | 2021-06-02 | 1        | 1      |
| _sample1458   | 1,0        | 256  | 256  | 2021-06-02 | 1        | 2      |
| ...           | ...        | ...  | ...  | ...        | ...      | ...    |

- **`Name`**: A unique identifier for each sample.
- **`Importance`**: Importance level.
- **`PosX` and `PosY`**: Size of the image [**TO BE CONFIRMED**]
- **`Date`**: Date of the ensemble forecast.
- **`LeadTime`**: Lead time in hours.
- **`Member`**: Member index within the ensemble.

This metadata file plays a crucial role in loading the dataset efficiently and ensuring the proper association of each sample with its corresponding attributes. Please update the file path in your code to reflect the location of your metadata CSV file.
## Training Experiments

Experiments are run with the [expe_init.py](https://github.com/flyIchtus/styleganPNRIA/blob/main/expe_init.py) file, which reads three configuration files found [here](https://github.com/flyIchtus/styleganPNRIA/tree/main/gan/configs).

### Description of configuration files:

### `main.yaml`

#### Experiment Initialization

- **data_dir**: Path to the directory containing input data.
- **output_dir**: Path to the directory where the experiment outputs will be stored.
- **config_dir**: Path to the directory containing configuration files. Example:
- **id_file**: Relative path to the CSV file containing labels, relative to data_dir.
- **SET_NUM**: Identifier for the experiment set.
- **max_relaunch**: Number of times to relaunch the experiment after timeout.
- **auto_relaunch**: Set to True for automatic relaunch. Manual relaunches (e.g., from pretrained) should have auto_relaunch=False to prevent overwriting previous results.
- **main_file**: Main Python script file for the experiment. Example: main_gan.py
- **slurm_file**: Script file called by Slurm's sbatch (expe_init_belenos). Relevant for the Belenos platform.
- **nb_gpus**: Number of GPUs to be used. Example: 4

#### Experiment Parameters (ensemble)

**General Parameters:**

- **total_steps**: Number of total steps for each experiment. Example: [500001]
- **epochs_num**: Number of epochs for each experiment. Example: [25]
- **pretrained_model**: Step indices for pretrained models. Set to -1 if no pretrained model. Example: [108000]

**Generator and Discriminator Configuration:**

- **var_names**: Variable names for used training. Example: ["[rr,u,v,t2m]"]
- **batch_size**: Real batch size is batch_size * N, where N is the number of GPUs. Example: [8]
- **lr_D**: Learning rate for the discriminator. Example: [0.002]
- **lr_G**: Learning rate for the generator. Example: [0.002]
- **g_channels**: Number of generator channels. Example: [3]
- **d_channels**: Number of discriminator channels. Example: [3]
- **path_batch_shrink**: Batch shrinkage factors for the path regularization. Example: [2]
- **tanh_output**: Use of tanh output. Example: [True]

**StyleGAN Configuration:**

- **model**: Model name. Example: ['stylegan2']
- **train_type**: Training type. Example: ['stylegan']
- **latent_dim**: Dimension of the latent space. Example: [512]
- **use_noise**: Use of noise injection. Example: [False]

**Database Parameters:**

- **crop_indexes**: Crop indexes for input AROME forecasts. Example: ["[0,256,0,256]"]
- **crop_size**: Crop size. Example: ["[256,256]"]
- **full_size**: Full size of input AROME forecasts. Example: ["[256,256]"]

**Configuration Files for dataset handling and scheduler config:**

- **dataset_handler_config**: Relative paths to dataset handler configuration files. Example: ['dataset_handler_config.yaml']
- **scheduler_config**: Relative paths to scheduler configuration files. Example: ['scheduler_config.yaml']


### dataset_handler_config.yaml

The configuration file includes various settings for preprocessing and normalization of the precipitation variable. Below are the details of each parameter:

- **stat_folder**: Path to the folder where statistical files are stored. Example: '"/stat_files"'

- **stat_version**: Name of the statistical file. The actual file name would be, for example, `mean_[stat_version]_log_ppx.npy`. Example: '"rr"'

#### Pour la variable rr (For_rr)
- **log_transform_iteration**: Number of times the log transformation is applied to the variable 'rr'. Example: '1'

- **symetrization**: Whether symmetrization is applied to the variable 'rr'. Example: 'False'

- **gaussian_std**: Threshold between rain and no rain to add Gaussian noise where 'rr < gaussian_std'. Example: '0'

#### Normalization

- **type**: Type of normalization to be applied. Choose between '"mean"', '"minmax"', or '"None"'. Example: '"minmax"'

- **per_pixel**: Whether normalization is applied with global values to each pixel or specific pixel values. Example: 'False'

    If `per_pixel` is `True`, the following options are used. For the 'rr' variable:

    - **blur_iteration**: The number of times a Gaussian convolution is applied to the grid containing the max/min/mean/max_std. Example: '1'




### Launching example

```python
python3 expe_init.py --config_file exemple_config/main.yaml # Path relative to `config_path`
}
``` 
will launch an experiment with the configuration given by the .yaml files.


### Output Directory Structure

The output files will be written in `output_dir`.

| Path | Description |
| --- | --- |
| [Set_1](output_dir/Set_1/Memo_readme.csv) | Main output directory containing the experiment set. |
|&ensp;&ensp;&boxvr;&nbsp; [Memo_readme.csv](./Set_1/Memo_readme.csv) | CSV file containing a summary of the parameter configuration used for the experiment set. |
|&ensp;&ensp;&boxvr;&nbsp;  [slurm-X.out](./Set_1/slurm-17812537.out) | Slurm output file will always be found in output_dir/Set_X/ |
|&ensp;&ensp;&boxur;&nbsp;  [stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False) | Directory containing StyleGAN experiment results |
|&ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; [Instance_X/](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_X) | Subdirectory for Instance X of the experiment |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [log/](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/log) | Log directory containing metrics calculated on-the-fly. |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [GAN_metrics_graph.png](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/log/GAN_metrics_graph.png) | Figure containing a metric summary of the training. |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; [metrics.csv](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/log/metrics.csv) | CSV file with metrics data for the training. |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [models/](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/models) | Directory for saving trained models at different steps |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; [X.pt](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/models/000000.pt) | Model checkpoint at Step X |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxvr;&nbsp; [ReadMe_1.txt](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/ReadMe_1.txt) | ReadMe file for Instance 1 |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; [samples/](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/samples) | Directory containing generated samples at different steps |
|&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&boxur;&nbsp; [_Fsample_0_X.npy](./Set_1/stylegan2_stylegan_dom_256_lat-dim_512_bs_8_0.002_0.002_ch-mul_2_vars_rr_u_v_t2m_noise_False/Instance_1/samples/_Fsample_0_1.npy) | Sample data at Step X |


## INVERSION TO THE LATENT SPACE

Once a skilled Generator is obtained, one can invert real AROME ensemble forecasts to the latent space using [main_inversion](https://github.com/flyIchtus/styleganPNRIA/blob/main/main_inversion.py). The inversion is configurated with the following parser parameters:

#### Directory Paths

- **`--ckpt_dir`**: Path to the checkpoint directory containing the pre-trained StyleGAN model.  
  - *Default*: ``

- **`--real_data_dir`**: Path to the directory containing real data used for inversion.  
  - *Default*: ``

- **`--output_dir`**: Path to the directory where the inversion results will be stored.  
  - *Default*: ``

- **`--pack_dir`**: Path to the directory where the real normalized ensembles that are inverted are stored.  
  - *Default*: ``

- **`--mean_file`**: File containing mean values for normalization.  
  - *Default*: ``

- **`--max_file`**: File containing max values for normalization.  
  - *Default*: ``

- **`--device`**: Device to run the inversion on (e.g., 'cuda:0').  
  - *Default*: `'cuda:0'`

#### Inversion Parameters. For more details, check the original [StyleGAN2 paper](https://arxiv.org/abs/1912.04958) and the [implementation](https://github.com/rosinality/stylegan2-pytorch) this repository is based on.

- **`--lr_rampup`**: Duration of the learning rate warmup.  
  - *Default*: `0.05`

- **`--lr_rampdown`**: Duration of the learning rate decay.  
  - *Default*: `0.25`

- **`--lr`**: Learning rate for optimization.  
  - *Default*: `0.1`

- **`--noise`**: Strength of the noise level.  
  - *Default*: `0.005`

- **`--noise_ramp`**: Duration of the noise level decay.  
  - *Default*: `0.75`

- **`--invstep`**: Number of optimization iterations.  
  - *Default*: `1000`

- **`--var_indices`**: List of variable indices to invert (e.g., [1,2,3]). Highly dependant on the shape of the samples of the dataset.  
  - *Default*: `[1,2,3]`

- **`--Shape`**: Size of the samples as a tuple (channels, height, width).  
  - *Default*: `(3,256,256)`

- **`--noise_regularize`**: Weight of the noise regularization during inversion.  
  - *Default*: `10e5`

- **`--loss`**: Type of loss function used (options: 'mse' or 'mae').  
  - *Default*: `'mse'`

- **`--loss_intens`**: Weight of the pixel loss.  
  - *Default*: `1.0`

- **`--inv_checkpoints`**: List of optimization steps to save results.  
  - *Default*: `[200,400,600,800,1000]`

#### Data Control for Inversion

- **`--dates_file`**: CSV file containing dates for inversion.  
  - *Default*: `'Large_lt_test_labels.csv'`

- **`--date_start`**: Start date for inversion in the format 'YYYY-MM-DD'.  
  - *Default*: `'2021-06-01'`

- **`--date_stop`**: Stop date for inversion in the format 'YYYY-MM-DD'.  
  - *Default*: `'2021-15-11'`

- **`--leadtimes`**: List of lead times for inversion.  
  - *Default*: `[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]`

## GENERATION OF GAN-ENRICHED ENSEMBLES

Once a skilled Generator is obtained real ensemble members have successfully been inverted to the latent space, one can enrich this ensembles using [main_perturbation.py](https://github.com/flyIchtus/styleganPNRIA/blob/main/main_perturbation.py)

#### Directory Paths

- **`--ckpt_dir`**: Path to the directory containing the pre-trained StyleGAN checkpoint.  
  - *Default*: `''`

- **`--real_data_dir`**: Path to the directory containing the full dataset.  
  - *Default*: `''`

- **`--data_dir`**: Path to the data directory containing the inversed ensembles.  
  - *Default*: `''`

- **`--output_dir`**: Path to the directory where the gan-enriched ensembles will be stored.  
  - *Default*: `''`

- **`--pack_dir`**: Path to the directory where the normalized real ensembles are stored.  
  - *Default*: `''`

- **`--mean_file`**: File containing mean values for normalization.  
  - *Default*: `''`

- **`--max_file`**: File containing max values for normalization.  
  - *Default*: `''`

- **`--var_indices`**: List of variable indices to be used.  
  - *Default*: `[1,2,3]`

- **`--Shape`**: Size of the samples as a tuple (channels, height, width).  
  - *Default*: `(3,256,256)`

- **`--N_samples`**: Ensemble size of the generated ensembles.  
  - *Default*: `120`

- **`--inv_step`**: Which step of the inversion process of the real ensembles should be used.  
  - *Default*: `1000`

#### Perturbation Parameters

- **`--sample_rule`**: Perturbation method used for generating new ensembles (options: 'random', 'normal', 'w', 'extrapolation').  
  - *Default*: `'random'`

- **`--style_indices`**: Which vectors of the latent code should be perturbed.  
  - *Default*: `'[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]'`

- **`--conditioning_members`**: Number of members used to generate perturbed ensembles (Max = 16).
  - *Default*: `16`

## Data Control for Perturbation

- **`--dates_file`**: CSV file containing dates for the ensembles to be enriched.  
  - *Default*: `''`

- **`--date_start`**: Start date for the ensembles to be enriched in the format 'YYYY-MM-DD'.  
  - *Default*: `'2020-07-01'`

- **`--date_stop`**: Stop date for the ensembles to be enriched in the format 'YYYY-MM-DD'.  
  - *Default*: `'2020-12-31'`

- **`--leadtimes`**: List of lead times for perturbation or inversion.  
  - *Default*: `[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45]`

#### Additional Configuration

- **`--runtime_metrics`**: Flag to enable the collection of runtime metrics.  
  - *Default*: `False`

## TODO :

Test various configs to validate 'test_merge_rr' branch
Merge 'test_merge_rr' branch to main
Prune (severely) the other branches
Do some cleaning with old, unused pieces of code
Commit some code from belenos about optuna optimization and perturbation pca (create new branch, push, test and merge)
