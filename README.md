
# Context

A new proposal to generate ensemble predictions matching the AROME-EPS dataset. The goal is to enrich the AROME-EPS dataset by generating samples mimicking the training data (i.e. to re-sample data from the latent distribution).  

The StyleGAN architecture has been the starting point of the  work, given encouraging results recently obtained by Brochet et al. [Multivariate emulation of convective-scale numerical weather predictions with generative adversarial networks]https://doi.org/10.1175/AIES-D-23-0006.1 and [Enriching Operational High-Resolution Ensemble Forecasts with StyleGAN-2] https://doi.org/10.1175/AIES-D-24-0058.1.

Let us now recall the two main configurations explored in this work, referred to as unconditional and conditional generations. In the unconditional setting, the machine learning method generates random samples from the training distribution. On the other hand, conditional generation aims at producing samples consistent with a given distribution. The latter is the configuration retained to super-sample NWP ensembles: in that case, the generated members should be consistent, and thus conditioned on, existing NWP members. The conditional setup with StyleGAN is a two-step procedure. The first step is called inversion, and consists in projecting existing NWP members in the latent space of the StyleGAN. The second step performs the conditional generation through latent space edition: new members are obtained by perturbing the existing ones in the latent space.

# StyleGAN4AROME

Even though several models are available for training, the current research is focusing on stylegan2 network (see the [original implementation](https://github.com/NVlabs/stylegan2) and the [pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch). 
A non-exhaustive diagram representing the global architecture is available on [Google Drive](https://drive.google.com/file/d/12Yidj0SBGblODHQIHi9Gf1WzNTqLoiJq/view?usp=sharing).  
Most of the core code is taken as is from [Rosinality's stylegan2-pytorch github page](https://github.com/rosinality/stylegan2-pytorch).


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

!WARNING! Don't forget to change the paths in config file.

## INVERSION TO THE LATENT SPACE

Once a skilled Generator is obtained, one can invert real AROME ensemble forecasts to the latent space using [main_inversion](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto//main_inversion.py).

The inversion is configurated with the parser parameters described in ** scripts_examples/init_inversion.sh **

For more details, check the original [StyleGAN2 paper](https://arxiv.org/abs/1912.04958) and the [implementation](https://github.com/rosinality/stylegan2-pytorch) this repository is based on.


## GENERATION OF GAN-ENRICHED ENSEMBLES

Once a skilled Generator is obtained real ensemble members have successfully been inverted to the latent space, one can enrich this ensembles using [main_perturbation.py](https://github.com/destination-earth-digital-twins/DE371_StyleGAN/tree/wp1_refacto//main_perturbation.py)

The perturbation is configurated with the parser parameters described in ** scripts_examples/init_perturbation.sh **


The files containing the values for normalization can be generated in the subfolder **preprocessing/Preprocess_datas_IS_split**.  

Each folder presented here includes its own description file.
