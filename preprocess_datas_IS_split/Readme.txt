To be completed to explain how to train gan on meluxina, how to do importance sampling, talk about observations, and how to invert and generate or plot GAN samples. 
Before training a dataset containing precipitation, it is important to use preferential sampling (PS) --> importance sampling. 
As summarised below, this allows us to avoid over-representation of samples without precipitation. 

What is importance sampling? The idea is to rank the data used
the data used by assigning them an ‘importance’ in terms of their contribution, for example, to
total variance of the dataset. The most important data are selected with a higher probability and used a posteriori.
and used a posteriori to train the network. In short, this is used to remove samples that 
contain little or no precipitation to force the network to learn to reproduce precipitation. 

To do this, we need to pre-process the dataset, to assemble our samples into Gigafiles (batches):
                        ‘pre_pro_for_is.py’

Then we need to use the file: ‘called/process_is.py’ to apply the IS. 

We're going to repeat this process n times, which is what we call bootstrapping, to obtain several csv files resulting from different EPs. 

We will then assemble these different csv files to enlarge our train set: ‘bootrstaps.py’.

Finally, we divide the dataset into train/test/valid set.

