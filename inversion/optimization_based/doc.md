# Optimization based inversion

This part of the code is dedicated to optimization based inversion of StyleGAN2. (see the [original implementation](https://github.com/NVlabs/stylegan2) and the [pytorch implementation](https://github.com/NVlabs/stylegan2-ada-pytorch).
The goal is to project the AROME physical ensemble to StyleGAN's latent space ensemble by optimizing the latent space.
All the default parameters can be found in ```main_inversion.py```.
To launch this part of the code, run the following command.

```
python3 main_inversion.py --inversion_type='optimization'
```
Author: Victor Sanchez
