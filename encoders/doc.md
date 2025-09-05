# Encoder for StyleGAN's Inversion

This part of the code is dedicated to the encoder used for StyleGAN's inverion.
The code is mainly inspired from previous work by [Alaluf et al.](https://arxiv.org/abs/2104.02699) and its [pytorch implementation](https://github.com/yuval-alaluf/restyle-encoder). It contains other recent work detailled as follow.

## e4e 
From [Designing an encoder for stylegan image manipulation](https://dl.acm.org/doi/pdf/10.1145/3450626.3459838?casa_token=zlp7L4Bwz-oAAAAA:texWCEXVCSuFXRiJdB2wrUF39fBXAMZ1xkEklghbIFxvdPDAOR5hfk0BBeu-L5So_0WEJDT6t2AmuQ) by Tov et al.


To launch a training of e4e encoder, run the following command:
```
python3 encoders/train_e4e.py'
```

## featureStyle 
From [Feature-style encoder for style-based gan inversion](https://arxiv.org/pdf/2202.02183) by Yao et al.

To launch a training of Feature-style encoder, run the following command:
```
python3 encoders/train_feature_style.py'
```

## HyperStyle 
From [Hyperstyle: Stylegan inversion with hypernetworks for real image editing](http://openaccess.thecvf.com/content/CVPR2022/papers/Alaluf_HyperStyle_StyleGAN_Inversion_With_HyperNetworks_for_Real_Image_Editing_CVPR_2022_paper.pdf) by Alaluf et al.

To launch a training of HyperStyle encoder, run the following command:
```
python3 encoders/train_hyper_style.py'
```

## inDomain 
From [In-domain gan inversion for faithful reconstruction and editability](https://arxiv.org/pdf/2004.00049) by Zhu et al.

To launch a training of HyperStyle encoder, run the following command:
```
python3 encoders/train_in_domain.py'
```

## pSp 
From [Encoding in style: a stylegan encoder for image-to-image translation](http://openaccess.thecvf.com/content/CVPR2021/papers/Richardson_Encoding_in_Style_A_StyleGAN_Encoder_for_Image-to-Image_Translation_CVPR_2021_paper.pdf) by Richardson et al.

To launch a training of pSp encoder, run the following command:
```
python3 encoders/train_pSp.py'
```

## restyle-e4e 
From [Restyle: A residual-based stylegan encoder via iterative refinement](http://openaccess.thecvf.com/content/ICCV2021/papers/Alaluf_ReStyle_A_Residual-Based_StyleGAN_Encoder_via_Iterative_Refinement_ICCV_2021_paper.pdf) by Alaluf et al.


To launch a training of Restyle e4e encoder, run the following command:
```
python3 encoders/train_restyle_e4e.py'
```

## restyle-pSp
From [Restyle: A residual-based stylegan encoder via iterative refinement](http://openaccess.thecvf.com/content/ICCV2021/papers/Alaluf_ReStyle_A_Residual-Based_StyleGAN_Encoder_via_Iterative_Refinement_ICCV_2021_paper.pdf) by Alaluf et al.


To launch a training of Restyle pSp encoder, run the following command:
```
python3 encoders/train_restyle_pSp.py'
```

Author: Victor Sanchez
