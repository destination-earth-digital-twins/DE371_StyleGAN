TO COMPLETE TO EXPLAIN HOW TRAIN A GAN IN MEKLUXINA, HOW TO DO IMPORTANCE SAMPLING , INVERSION PROCESS AND PERTURBATION 

Before training a dataset containing precipitation data, it is important to use preferential sampling (EP) → importance sampling.

What is importance sampling? The idea is to prioritize the data used by assigning them an "importance" based on their contribution to, for example, the total variance of the dataset. The most important data points are selected with a higher probability and later used for training the network. In summary, it helps to eliminate samples that contain little or no precipitation, forcing the network to learn to replicate precipitation patterns.

To achieve this, we need to pre-process the dataset to group our samples into Gigafile (batches):
"pre_pro_for_is.py"

Then, we need to use the file: "called/process_is.py" to apply the importance sampling (EP).

We will repeat this process multiple times, which is called bootstrap, to obtain several CSV files resulting from different EPs.

Next, we will combine these different CSVs to enlarge our training set: "bootstraps.py"

Finally, we divide the dataset into training, testing, and validation sets.