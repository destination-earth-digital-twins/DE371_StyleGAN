# Hybrid based inversion

This part of the code is dedicated to hybrid based inversion of StyleGAN2.
The goal is to project the AROME physical ensemble to StyleGAN's latent space ensemble by optimizing the latent space starting from a latent space projection obtained with a trained encoder.
All the default parameters can be found in ```main_inversion.py```.
To launch this part of the code, run the following command.

```
python3 main_inversion.py --inversion_type='hybrid'
```
Author: Victor Sanchez
