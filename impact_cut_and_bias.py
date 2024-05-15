import pickle
import numpy as np
import matplotlib.pyplot as plt

expes = {'11111111111111_False_1.0' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_1.0/", 'metrics.p', 'full_pca'],
         '11111111111110_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_1.0/", 'metrics.p', 'pca_13'],
         '11111111111100_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_1.0/", 'metrics.p', 'pca_12'],
         '11111111111000_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_11'],
         '11111111110000_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_10'],
         '11111111100000_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_9'],
         '11111111000000_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_8'],
         '11111110000000_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_7'],
         '11111100000000_False_1.0': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_6'],
         '11111000000000_False_1.0': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_5'],
         '11110000000000_False_1.0': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_4'],
         '11100000000000_False_1.0': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_3'],
         '11000000000000_False_1.0': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_2'],
         '10000000000000_False_1.0': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'pca_1'],
         '00000000000000_False_1.0': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0/", 'metrics.p', 'full_random'],

         '11111111111111_True_1.0' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_True_1.0/", 'metrics.p', 'full_pca'],
         '11111111111110_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_True_1.0/", 'metrics.p', 'pca_13'],
         '11111111111100_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_True_1.0/", 'metrics.p', 'pca_12'],
         '11111111111000_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_11'],
         '11111111110000_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_10'],
         '11111111100000_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_9'],
         '11111111000000_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_8'],
         '11111110000000_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_7'],
         '11111100000000_True_1.0': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_6'],
         '11111000000000_True_1.0': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_5'],
         '11110000000000_True_1.0': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_4'],
         '11100000000000_True_1.0': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_3'],
         '11000000000000_True_1.0': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_2'],
         '10000000000000_True_1.0': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'pca_1'],
         '00000000000000_True_1.0': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_1.0/", 'metrics.p', 'full_random'],

        
         '11111111111111_False_0.1' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_0.1/", 'metrics.p', 'full_pca'],
         '11111111111110_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_0.1/", 'metrics.p', 'pca_13'],
         '11111111111100_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_0.1/", 'metrics.p', 'pca_12'],
         '11111111111000_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_11'],
         '11111111110000_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_10'],
         '11111111100000_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_9'],
         '11111111000000_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_8'],
         '11111110000000_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_7'],
         '11111100000000_False_0.1': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_6'],
         '11111000000000_False_0.1': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_5'],
         '11110000000000_False_0.1': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_4'],
         '11100000000000_False_0.1': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_3'],
         '11000000000000_False_0.1': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_2'],
         '10000000000000_False_0.1': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'pca_1'],
         '00000000000000_False_0.1': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.1/", 'metrics.p', 'full_random'],

         '11111111111111_False_0.5' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_0.5/", 'metrics.p', 'full_pca'],
         '11111111111110_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_0.5/", 'metrics.p', 'pca_13'],
         '11111111111100_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_0.5/", 'metrics.p', 'pca_12'],
         '11111111111000_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_11'],
         '11111111110000_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_10'],
         '11111111100000_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_9'],
         '11111111000000_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_8'],
         '11111110000000_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_7'],
         '11111100000000_False_0.5': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_6'],
         '11111000000000_False_0.5': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_5'],
         '11110000000000_False_0.5': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_4'],
         '11100000000000_False_0.5': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_3'],
         '11000000000000_False_0.5': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_2'],
         '10000000000000_False_0.5': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_1'],
         '00000000000000_False_0.5': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'full_random'],

         '11111111111111_False_0.5_False' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_0.5/", 'metrics.p', 'full_pca'],
         '11111111111110_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_0.5_False/", 'metrics.p', 'pca_13'],
         '11111111111100_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_12'],
         '11111111111000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_11'],
         '11111111110000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_10'],
         '11111111100000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_9'],
         '11111111000000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_8'],
         '11111110000000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_7'],
         '11111100000000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_6'],
         '11111000000000_False_0.5_False': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_5'],
         '11110000000000_False_0.5_False': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_4'],
         '11100000000000_False_0.5_False': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_3'],
         '11000000000000_False_0.5_False': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_2'],
         '10000000000000_False_0.5_False': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'pca_1'],
         '00000000000000_False_0.5_False': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5_False/", 'metrics.p', 'full_random'],

         '11111111111111_False_2.0' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_2.0/", 'metrics.p', 'full_pca'],
         '11111111111110_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_2.0/", 'metrics.p', 'pca_13'],
         '11111111111100_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_2.0/", 'metrics.p', 'pca_12'],
         '11111111111000_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_11'],
         '11111111110000_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_10'],
         '11111111100000_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_9'],
         '11111111000000_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_8'],
         '11111110000000_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_7'],
         '11111100000000_False_2.0': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_6'],
         '11111000000000_False_2.0': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_5'],
         '11110000000000_False_2.0': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_0.5/", 'metrics.p', 'pca_4'],
         '11100000000000_False_2.0': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_3'],
         '11000000000000_False_2.0': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_2'],
         '10000000000000_False_2.0': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'pca_1'],
         '00000000000000_False_2.0': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_2.0/", 'metrics.p', 'full_random'],

         '11111111111111_True_0.5' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_True_0.5/", 'metrics.p', 'full_pca'],
         '11111111111110_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_True_0.5/", 'metrics.p', 'pca_13'],
         '11111111111100_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_True_0.5/", 'metrics.p', 'pca_12'],
         '11111111111000_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_11'],
         '11111111110000_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_10'],
         '11111111100000_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_9'],
         '11111111000000_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_8'],
         '11111110000000_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_7'],
         '11111100000000_True_0.5': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_6'],
         '11111000000000_True_0.5': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_5'],
         '11110000000000_True_0.5': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_4'],
         '11100000000000_True_0.5': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_3'],
         '11000000000000_True_0.5': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_2'],
         '10000000000000_True_0.5': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'pca_1'],
         '00000000000000_True_0.5': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_True_0.5/", 'metrics.p', 'full_random'],
         

         '11111111111111_False_1.0_True' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_1.0_True/", 'metrics.p', 'full_pca'],
         '11111111111110_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_1.0_True/", 'metrics.p', 'pca_13'],
         '11111111111100_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_12'],
         '11111111111000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_11'],
         '11111111110000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_10'],
         '11111111100000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_9'],
         '11111111000000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_8'],
         '11111110000000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_7'],
         '11111100000000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_6'],
         '11111000000000_False_1.0_True': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_5'],
         '11110000000000_False_1.0_True': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_4'],
         '11100000000000_False_1.0_True': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_3'],
         '11000000000000_False_1.0_True': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_2'],
         '10000000000000_False_1.0_True': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'pca_1'],
         '00000000000000_False_1.0_True': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_1.0_True/", 'metrics.p', 'full_random'],

         '11111111111111_False_4.3397_True' : ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']_False_4.3397_True/", 'metrics.p', 'full_pca'],
         '11111111111110_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0']_False_4.3397_True/", 'metrics.p', 'pca_13'],
         '11111111111100_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_12'],
         '11111111111000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_11'],
         '11111111110000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_10'],
         '11111111100000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_9'],
         '11111111000000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_8'],
         '11111110000000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_7'],
         '11111100000000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_6'],
         '11111000000000_False_4.3397_True': ["mix_['1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_5'],
         '11110000000000_False_4.3397_True': ["mix_['1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_4'],
         '11100000000000_False_4.3397_True': ["mix_['1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_3'],
         '11000000000000_False_4.3397_True': ["mix_['1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_2'],
         '10000000000000_False_4.3397_True': ["mix_['1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'pca_1'],
         '00000000000000_False_4.3397_True': ["mix_['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']_False_4.3397_True/", 'metrics.p', 'full_random'],
         }

def get_metric_mean(dic, metric):
    liste = list(dic.values())
    if type(liste[0][metric])==dict:
        final = {'real':np.zeros((3,)), 'fake':np.zeros((3,))}
        for v in liste:
            final['real']+=v[metric]['real']
            final['fake']+=v[metric]['fake']
        
        final['real'] /= float(len(liste))
        final['fake'] /= float(len(liste))
    else:
        final = np.zeros((3,))
        for v in liste:
            final+=v[metric]
        final /= float(len(liste))

    return final

def get_metric_std(dic, metric):

    metric_mean = get_metric_mean(dic, metric)
    metric_square_mean = get_metric_mean(get_func('square', lambda a : a **2, dic, metric),metric+'square')

    return np.sqrt(metric_square_mean - np.array(metric_mean)**2)


def get_metric_values(dic, metric):
    liste = dic.values()
    dic0 = {metric : {'real':0.0, 'fake':0.0}}
    for v in liste:
        dic0[metric]['real']+=v[metric]['real']
        dic0[metric]['fake']+=v[metric]['fake']
    
    dic0[metric]['real'] /= float(len(liste))
    dic0[metric]['fake'] /= float(len(liste))

    return dic0

def get_func(func_name, func,dic,metric):
    print(func_name, metric)
    dic0 = {}
    for k in dic.keys():
        #print(k, metric, func_name)
        dic0[k] = {}
        if type(dic[k][metric])==dict:
            dic0[k][metric+func_name] = func(dic[k][metric]['real'], dic[k][metric]['fake'])
        else:
            dic0[k][metric+func_name] = func(dic[k][metric])
    return dic0


def plot_all_mean(metric1, metric2, funcname1, funcname2, list_expes, funcs, variables=['u','v','t2m'],scale=1.0, root_dir = './'):
    dic_expes = {}
    mean_metric_1 = []
    std_metric_1 = []
    std_metric_2 = []
    mean_metric_2 = []
    for e_idx, expe in enumerate(list_expes):
        #print(e_idx, expe)
        name = expes[expe][-1]
        dic_expes[name] = pickle.load(open(root_dir + expes[expe][0] + 'log/' + expes[expe][1],'rb'))
        #print(dic_expes[name].keys())
        dic1 = get_func(funcname1, funcs[funcname1], dic_expes[name], metric1)
        dic2 = get_func(funcname2, funcs[funcname2], dic_expes[name], metric2)
        
        mean_metric_1.append(get_metric_mean(dic1, metric1+funcname1))
        std_metric_1.append(get_metric_std(dic1,metric1+funcname1))
        
        mean_metric_2.append(get_metric_mean(dic2, metric2+funcname2))
        std_metric_2.append(get_metric_std(dic2,metric2+funcname2))

    #print(len(mean_metric_1),type(mean_metric_1))

    for var_idx in range(len(variables)):
        fig, ax = plt.subplots(figsize=(6,6))
        color = 'tab:blue'
        ax.set_ylabel(metric1,color=color)
        ax.tick_params(axis='y', labelcolor=color)
        plt.xticks(ticks = np.arange(len(list_expes)) ,labels = list(dic_expes.keys()), rotation = 'vertical')
        ax.plot(range(len(list_expes)), np.array(mean_metric_1)[:,var_idx], 'bo-', label=metric1)
        ax.fill_between(range(len(list_expes)), 
                        np.array(mean_metric_1)[:,var_idx] - np.array(std_metric_1)[:,var_idx],
                        np.array(mean_metric_1)[:,var_idx] + np.array(std_metric_1)[:,var_idx], alpha=0.3,color='blue')
        secax = ax.twinx()
        color = 'tab:red'
        secax.set_ylabel(metric2,color=color)
        secax.tick_params(axis='y', labelcolor=color)
        secax.plot(range(len(list_expes)), np.array(mean_metric_2)[:,var_idx], 'ro-', label=metric2)
        secax.fill_between(range(len(list_expes)), 
                        np.array(mean_metric_2)[:,var_idx] - np.array(std_metric_2)[:,var_idx],
                        np.array(mean_metric_2)[:,var_idx] + np.array(std_metric_2)[:,var_idx], alpha=0.3,color='red')
        #ax.legend()
        #secax.legend()
        plt.grid()
        print(dic_expes.keys(), list(dic_expes.keys()))
        #plt.show()
        plt.savefig(f"{root_dir}{metric1}_{funcname1}_{metric2}_{funcname2}_{variables[var_idx]}_{scale}.png")
        plt.close()

def compare_expe_type_mean(metric1, metric2, funcname1, funcname2, list_list_expes, list_labels, funcs, variables=['u','v','t2m'], root_dir = './', suffix='mult'):

    forms = ['o','-','+','d']

    dic_expes = {}
    mean_metric_1 = []

    std_metric_1 = []

    std_metric_2 = []

    mean_metric_2 = []

    for type_idx in range(len(list_list_expes)):
        mm1 = []
        mm2 = []
        std1 = []
        std2 = []
        for e_idx, expe in enumerate(list_list_expes[type_idx]):
            #print(e_idx, expe)
            name = expes[expe][-1]
            dic_expes[name] = pickle.load(open(root_dir + expes[expe][0] + 'log/' + expes[expe][1],'rb'))
            #print(dic_expes[name].keys())
            dic1 = get_func(funcname1, funcs[funcname1], dic_expes[name], metric1)
            dic2 = get_func(funcname2, funcs[funcname2], dic_expes[name], metric2)
            
            mm1.append(get_metric_mean(dic1, metric1+funcname1))
            std1.append(get_metric_std(dic1,metric1+funcname1))
            
            mm2.append(get_metric_mean(dic2, metric2+funcname2))
            std2.append(get_metric_std(dic2,metric2+funcname2))
        mean_metric_1.append(mm1)
        mean_metric_2.append(mm2)

        std_metric_1.append(std1)
        std_metric_2.append(std2)

    #print(len(mean_metric_1),type(mean_metric_1))
    print(len(mean_metric_1), len(mean_metric_1[0]), np.array(mean_metric_1[0]).shape)
    for var_idx in range(len(variables)):
        fig, ax = plt.subplots(figsize=(6,6))
        color = 'tab:blue'
        ax.set_ylabel(metric1,color=color)
        ax.tick_params(axis='y', labelcolor=color)
        plt.xticks(ticks = np.arange(len(list_list_expes[0])),labels = list(dic_expes.keys()), rotation = 'vertical')
        
        for type_idx, type_exp in enumerate(list_list_expes):
            print(type_idx)
            fo = forms[type_idx]
            ax.plot(range(len(type_exp)), np.array(mean_metric_1[type_idx])[:,var_idx],f'b{fo}-', label=list_labels[type_idx])
            ax.fill_between(range(len(type_exp)), 
                        np.array(mean_metric_1[type_idx])[:,var_idx] - np.array(std_metric_1[type_idx])[:,var_idx],
                        np.array(mean_metric_1[type_idx])[:,var_idx] + np.array(std_metric_1[type_idx])[:,var_idx], alpha=0.3, color='blue')
        secax = ax.twinx()
        color = 'tab:red'
        secax.set_ylabel(metric2,color=color)
        secax.tick_params(axis='y', labelcolor=color)
        for type_idx, type_exp in enumerate(list_list_expes):
            print(type_idx)
            fo=forms[type_idx]
            secax.plot(range(len(type_exp)), np.array(mean_metric_2[type_idx])[:,var_idx],f'r{fo}-', label=list_labels[type_idx])
            secax.fill_between(range(len(type_exp)), 
                        np.array(mean_metric_2[type_idx])[:,var_idx] - np.array(std_metric_2[type_idx])[:,var_idx],
                        np.array(mean_metric_2[type_idx])[:,var_idx] + np.array(std_metric_2[type_idx])[:,var_idx], alpha=0.3,color='red')
        ax.legend()
        #secax.legend()
        plt.grid()
        print(dic_expes.keys(), list(dic_expes.keys()))
        #plt.show()
        plt.savefig(f"{root_dir}{metric1}_{funcname1}_{metric2}_{funcname2}_{variables[var_idx]}_{suffix}.png")
        plt.close()

if __name__=="__main__":

    root_dir = '/scratch/work/brochetc/Exp_StyleGAN/'

    funcs = {'mae' : (lambda a,b : np.abs(a-b)), 
            'diff': (lambda a,b : a-b),
            'ratio' : (lambda a,b : a/b),
            '': lambda a : a}

    list_expes_01 = [ k for k in expes.keys() if '0.1' in k]
    list_expes_05_t = [ k for k in expes.keys() if '0.5' in k and 'True' in k]
    list_expes_05_f = [ k for k in expes.keys() if 'False_0.5' in k and 'False_0.5_False' not in k and 'False_0.5_True' not in k]
    list_expes_05_trunc = [ k for k in expes.keys() if 'False_0.5_False' in k]#
    
    list_expes_1 = [ k for k in expes.keys() if 'False_1.0' in k and 'False_1.0_True' not in k]
    list_expes_2 = [ k for k in expes.keys() if 'False_2.0' in k ]# 'False_1.0_True' not in k]
    #list_expes_renorm = [k for k in expes.keys() if 'False_1.0_True' in k]
    #list_expes_renorm_inflated = [k for k in expes.keys() if 'False_4.3397_True' in k]
    list_list_expes = [list_expes_05_f, list_expes_05_trunc]#, list_expes_1] list_expes_renorm_inflated, 
    list_labels = ['standard', 'truncated'] #['scale=0.5', "scale=1.0", 'scale=2.0'] #'Renorm Inflated', 
    print(list_list_expes)
    metric1 = 'Mean'
    funcname1 = 'mae'
    
    metric2 = 'Max'
    funcname2 = 'diff'

    compare_expe_type_mean(metric1, metric2, funcname1, funcname2,list_list_expes,list_labels, funcs,root_dir=root_dir, suffix='std_trunc')




    