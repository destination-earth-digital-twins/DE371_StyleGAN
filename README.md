# StyleGAN4AROME
A new proposal to generate ensemble predictions matching the AROME-EPS dataset. Even though several models are available for training, the current research is focusing on stylegan2 network (see the [original implementation](https://github.com/NVlabs/stylegan2) and the [pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch). The goal is to enrich the AROME-EPS dataset by generating samples mimicking the training data (i.e. to re-sample data from the latent distribution).  
A non-exhaustive diagram representing the global architecture is available on [Google Drive](https://drive.google.com/file/d/12Yidj0SBGblODHQIHi9Gf1WzNTqLoiJq/view?usp=sharing).  
Most of the core code is taken as is from [Rosinality's stylegan2-pytorch github page](https://github.com/rosinality/stylegan2-pytorch) and adapted to  run on Meteo France clusters. 
See the paper here: ??
Authors: C. Brochet, G. Moldovan, V. Sanchez, A. Bonamy

## The AROME-EPS Dataset

The dataset comprises 516 AROME ensemble forecasts covering the period from June 15th, 2020, to November 12th, 2021. Each ensemble forecast is composed of 16 members and includes lead times at 1-hour intervals, ranging up to 45 hours. It follows that [516x45x16=371520]() individual samples are available for training if each members of the enseble at a given lead time is considered individually.

The data is restricted to a region encompassing the south and center of France with a resolution of [256x256]. Four variables are here considered: the precipitation (rr) the horizontal (u) and vertical (v) components of the wind speed vector at 10 meters and the temperature at 2 meters (t2m). Each individual sample can be conceptualized as a tensor with 4 channels, a width of 256 and a height of 256 [4, 256, 256].

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

### Temporal downscaling in the StyleGAN latent space (additional code):
- `train_interpolator.py` - main script for training the temporal downscaling model
- `temporal_interpolation.py` - main script for performing the temporal interpolation using the trained model

`time_interpolation` source folder:
  - `dataset.py` - description of the `InterpolatorDataset` class
  - `models.py` - description of the interpolator model classes
  - `training.py` - description of interpolation model training routines

### Launching examples

Interpolator model training: 
`./init_train_interpolator.sh`

Temporal interpolation:
`./init_temporal_interpolation.sh`


