# StyleGAN4AROME

A new proposal to generate ensemble predictions matching the AROME-EPS dataset. Even though several models are available for training, the current research is focusing on stylegan2 network (see the [original implementation](https://github.com/NVlabs/stylegan2) and the [pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch). The goal is to enrich the AROME-EPS dataset by generating samples mimicking the training data (i.e. to re-sample data from the latent distribution).  
A non-exhaustive diagram representing the global architecture is available on [Google Drive](https://drive.google.com/file/d/12Yidj0SBGblODHQIHi9Gf1WzNTqLoiJq/view?usp=sharing).  
Most of the core code is taken as is from [Rosinality's stylegan2-pytorch github page](https://github.com/rosinality/stylegan2-pytorch).

# Context
The StyleGAN architecture has been the starting point of the DE_371 work, given encouraging results recently obtained by Brochet et al. [Multivariate emulation of convective-scale numerical weather predictions with generative adversarial networks]https://doi.org/10.1175/AIES-D-23-0006.1 and [Enriching Operational High-Resolution Ensemble Forecasts with StyleGAN-2] https://doi.org/10.1175/AIES-D-24-0058.1.

Let us now recall the two main configurations explored in this work, referred to as unconditional and conditional generations. In the unconditional setting, the machine learning method generates random samples from the training distribution. On the other hand, conditional generation aims at producing samples consistent with a given distribution. The latter is the configuration retained to super-sample NWP ensembles: in that case, the generated members should be consistent, and thus conditioned on, existing NWP members. The conditional setup with StyleGAN is a two-step procedure. The first step is called inversion, and consists in projecting existing NWP members in the latent space of the StyleGAN. The second step performs the conditional generation through latent space edition: new members are obtained by perturbing the existing ones in the latent space. 

# Repository Structure

| Path | Description |
| --- | --- |
|[DE371_StyleGAN](https://github.com/destination-earth-digital-twins/DE371_StyleGAN)|Root folder of the repository|
|&ensp;&ensp;&boxvr;&nbsp; [autoencoder](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/encoders)|Subfolder with auto-encoder-based StyleGAN Inversion approach |
|&ensp;&ensp;&boxvr;&nbsp; [encoders](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/encoders)|Subfolder with Encoder-based StyleGAN Inversion approach |
|&ensp;&ensp;&boxvr;&nbsp; [gan](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/gan)|Subfolder with StyleGAN code|
|&ensp;&ensp;&boxvr;&nbsp; [grandEnsemble](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/grandEnsemble)|Subfolder with grandEnsemble study|
|&ensp;&ensp;&boxvr;&nbsp; [inversion](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/inversion)|Subfolder with StyleGAN Inversion (Optimization, Encoder Hybrid) related file|
|&ensp;&ensp;&boxvr;&nbsp; [latent_analysis](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/latent_analysis)|Subfolder for latent analysis of StyleGAN|
|&ensp;&ensp;&boxvr;&nbsp; [perturbation](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/perturbation)|Subfolder with perturbation-related files|
|&ensp;&ensp;&boxvr;&nbsp; [plot_analysis](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/plot_analysis)|Main folder for plot analysis|
|&ensp;&ensp;&boxvr;&nbsp; [preprocessing](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/preprocessing)|Subfolder with preprocessing-related files|
|&ensp;&ensp;&boxvr;&nbsp; [scripts_examples](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/)|Main folder for scripts|
|&ensp;&ensp;&boxvr;&nbsp; [time_interpolation](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/)|folder to compute time interpolation between different generated samples|
|&ensp;&ensp;&boxvr;&nbsp; [container.def](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/container.def)|Apptainer configuration file|
|&ensp;&ensp;&boxvr;&nbsp; [Dockerfile](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/Dockerfile)|Docker configuration file|
|&ensp;&ensp;&boxvr;&nbsp; [main_gan.py](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/main_gan.py)|Script for main GAN operations|
|&ensp;&ensp;&boxvr;&nbsp; [main_inversion.py](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/main_inversion.py)|Script for inversion operations|
|&ensp;&ensp;&boxvr;&nbsp; [main_perturbation.py](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/main_perturbation.py)|Script for generation of GAN ensembles|
|&ensp;&ensp;&boxvr;&nbsp; [README.md](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/README.md)|Main repository documentation|
|&ensp;&ensp;&boxur;&nbsp; [requirements.txt](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/requirements.txt)|List of required Python packages|




## The AROME-EPS Dataset

The dataset comprises 516 AROME ensemble forecasts covering the period from June 15th, 2020, to November 12th, 2021. Each ensemble forecast is composed of 16 members and includes lead times at 1-hour intervals, ranging up to 45 hours. It follows that [516x45x16=371520]() individual samples are available for training if each members of the enseble at a given lead time is considered individually.

The data is restricted to a region encompassing the south and center of France with a resolution of [256x256]. Four variables are here considered: the precipitation (rr in mm/h) the horizontal (u) and vertical (v) in m/s components of the wind speed vector at 10 meters and the temperature at 2 meters (t2m)in K. Each individual sample can be conceptualized as a tensor with 4 channels, a width of 256 and a height of 256 [4, 256, 256].

To efficiently load and organize the dataset, a metadata CSV file is utilized. The file structure is illustrated below:

| Name          | Importance | PosX | PosY | Date       | LeadTime | Member |
|---------------|------------|------|------|------------|----------|--------|
| ...           | ...        | ...  | ...  | ...        | ...      | ...    |
| _sample1440   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 0      |
| _sample1441   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 1      |
| _sample1442   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 2      |
| _sample1443   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 3      |
| _sample1444   | 1,0        | 256  | 256  | 2021-06-02 | 0        | 4      |

- **`Name`**: A unique identifier for each sample.
- **`Importance`**: Importance level.
- **`PosX` and `PosY`**: Size of the image [**TO BE CONFIRMED**]
- **`Date`**: Date of the ensemble forecast.
- **`LeadTime`**: Lead time in hours.
- **`Member`**: Member index within the ensemble.

This metadata file plays a crucial role in loading the dataset efficiently and ensuring the proper association of each sample with its corresponding attributes. Please update the file path in your code to reflect the location of your metadata CSV file.

## Training Experiments

Experiments are run with the [main_gan.py](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto/main_gan.py) file


### Launching example

```python
  sbatch ./scripts_examples/init_training.sh
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

Once a skilled Generator is obtained, one can invert real AROME ensemble forecasts to the latent space using [main_inversion](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto//main_inversion.py). The inversion is configurated with the following parser parameters:

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
  - *Default*: `[0,1,2,3]`

- **`--Shape`**: Size of the samples as a tuple (channels, height, width).  
  - *Default*: `(4,256,256)`

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

Once a skilled Generator is obtained real ensemble members have successfully been inverted to the latent space, one can enrich this ensembles using [main_perturbation.py](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto//main_perturbation.py)

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
  - *Default*: `[0,1,2,3]`

- **`--Shape`**: Size of the samples as a tuple (channels, height, width).  
  - *Default*: `(4,256,256)`

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

The files containing the values for normalization can be generated in the subfolder **preprocessing/Preprocess_datas_IS_split**.  

Each folder presented here includes its own description file.
