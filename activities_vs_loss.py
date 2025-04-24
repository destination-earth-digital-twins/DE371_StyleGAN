import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

root_dir = '/project/scratch/p200177/DE_371/angeliquebonamy/test_scale_tune/vgg/sigmoid_stdmean_rruvt/'
#/project/scratch/p200177/DE_371/angeliquebonamy/test_scale_tune/vgg/sigmoid_quantile_rr_stdmean_uvt

colors = {0 : 'yellow', 8 : 'violet', 10: 'red', 12 : 'darkblue', 14 : 'darkgreen'}

scale_expes = {
    # 'cut=0_infl1.0' : ['interp_scale_pca_0_False_1.0_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',0,1.0],
    # 'cut=0_infl1.1' : ['interp_scale_pca_0_False_1.1_bias_ones_1.0_spread_1.0_ff_False_1000//Instance_1/',0,1.1],
    # 'cut=0_infl1.2' : ['interp_scale_pca_0_False_1.2_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',0,1.2],
    # 'cut=0_infl1.3' : ['interp_scale_pca_0_False_1.3_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',0,1.3],
    
    'cut=8_infl1.0' : ['interp_scale_pca_8_False_1.0_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',8,1.0],
    # 'cut=8_infl1.1' : ['interp_scale_pca_8_False_1.1_bias_ones_1.0_spread_1.0_ff_False_1000//Instance_1/',8,1.1],
    'cut=8_infl1.2' : ['interp_scale_pca_8_False_1.2_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',8,1.2],
    'cut=8_infl1.25' : ['interp_scale_pca_8_False_1.25_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_3/',8,1.25],
    'cut=8_infl1.3' : ['interp_scale_pca_8_False_1.3_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',8,1.3],
    'cut=8_infl1.5' : ['interp_scale_pca_8_False_1.5_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',8,1.5],

    'cut=10_infl1.0' : ['interp_scale_pca_10_False_1.0_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',10,1.0],
    # 'cut=10_infl1.1' : ['interp_scale_pca_10_False_1.1_bias_ones_1.0_spread_1.0_ff_False_1000//Instance_1/',10,1.1],
    'cut=10_infl1.2' : ['interp_scale_pca_10_False_1.2_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',10,1.2],
    'cut=10_infl1.25' : ['interp_scale_pca_10_False_1.25_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',10,1.25],
    'cut=10_infl1.3' : ['interp_scale_pca_10_False_1.3_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',10,1.3],
    'cut=10_infl1.5' : ['interp_scale_pca_10_False_1.5_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',10,1.5],

    # 'cut=12_infl1.0' : ['interp_scale_pca_12_False_1.0_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',12,1.0],
    # # 'cut=12_infl1.1' : ['interp_scale_pca_12_False_1.1_bias_ones_1.0_spread_1.0_ff_False_1000//Instance_1/',12,1.1],
    # 'cut=12_infl1.2' : ['interp_scale_pca_12_False_1.2_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',12,1.2],
    # # 'cut=12_infl1.3' : ['interp_scale_pca_12_False_1.3_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',12,1.3],
    # 'cut=12_infl1.5' : ['interp_scale_pca_12_False_1.5_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',12,1.5],

        
    # 'cut=14_infl1.0' : ['interp_scale_pca_14_False_1.0_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',14,1.0],
    # # 'cut=14_infl1.1' : ['interp_scale_pca_14_False_1.1_bias_ones_1.0_spread_1.0_ff_False_1000//Instance_1/',14,1.1],
    # 'cut=14_infl1.2' : ['interp_scale_pca_14_False_1.2_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',14,1.2],
    # # 'cut=14_infl1.3' : ['interp_scale_pca_14_False_1.3_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',14,1.3],
    # 'cut=14_infl1.5' : ['interp_scale_pca_14_False_1.5_bias_ones_1.0_spread_1.0_ff_False_1000/Instance_1/',14,1.5],

    }

attributes = {
#  'cut=0_infl1.0' : {'cut' : 0, 'inflt' :0.0},
#  'cut=0_infl1.1' : {'cut' : 0, 'inflt' :0.1},
#  'cut=0_infl1.2' : {'cut' : 0, 'inflt' :0.2},
#  'cut=0_infl1.3' : {'cut' : 0, 'inflt' :0.3},


 'cut=8_infl1.0' : {'cut' : 8, 'inflt' :0.0},
#  'cut=8_infl1.1' : {'cut' : 8, 'inflt' :0.1},
 'cut=8_infl1.2' : {'cut' : 8, 'inflt' :0.2},
  'cut=8_infl1.25' : {'cut' : 8, 'inflt' :0.25},
 'cut=8_infl1.3' : {'cut' : 8, 'inflt' :0.3},
 'cut=8_infl1.5' : {'cut' : 8, 'inflt' :0.5},

 
 'cut=10_infl1.0' : {'cut' : 10, 'inflt' :0.0},
#  'cut=10_infl1.1' :{'cut' : 10, 'inflt' :0.1},
 'cut=10_infl1.2' : {'cut' : 10, 'inflt' :0.2},
 'cut=10_infl1.25' : {'cut' : 10, 'inflt' :0.25},
 'cut=10_infl1.3' : {'cut' : 10, 'inflt' :0.3},
 'cut=10_infl1.5' : {'cut' : 10, 'inflt' :0.5},

 
#  'cut=12_infl1.0' : {'cut' : 12, 'inflt' :0.0},
# #  'cut=12_infl1.1' : {'cut' : 12, 'inflt' :0.1},
#  'cut=12_infl1.2' : {'cut' : 12, 'inflt' :0.2},
# #  'cut=12_infl1.3' : {'cut' : 12, 'inflt' :0.3},
#  'cut=12_infl1.5' : {'cut' : 12, 'inflt' :0.5},

 
#  'cut=14_infl1.0' :  {'cut' : 14, 'inflt' :0.0},
# #  'cut=14_infl1.1' :  {'cut' : 14, 'inflt' :0.1},
#  'cut=14_infl1.2' :  {'cut' : 14, 'inflt' :0.2},
# #  'cut=14_infl1.3' :  {'cut' : 14, 'inflt' :0.3},
#  'cut=14_infl1.5' :  {'cut' : 14, 'inflt' :0.5},

}

key_groups = [
# ['cut=0_infl1.0',
#  'cut=0_infl1.1',
#  'cut=0_infl1.2',
#  'cut=0_infl1.3'],

['cut=8_infl1.0',
#  'cut=8_infl1.1',
 'cut=8_infl1.2',
 'cut=8_infl1.25',
 'cut=8_infl1.3',
 'cut=8_infl1.5'],

['cut=10_infl1.0',
#  'cut=8_infl1.1',
 'cut=10_infl1.2',
 'cut=10_infl1.25',
 'cut=10_infl1.3',
 'cut=10_infl1.5'],

# ['cut=12_infl1.0',
# #  'cut=8_infl1.1',
#  'cut=12_infl1.2',
# #  'cut=8_infl1.3'],
#  'cut=12_infl1.5'],

# ['cut=14_infl1.0',
# #  'cut=8_infl1.1',
#  'cut=14_infl1.2',
# #  'cut=8_infl1.3'],
#  'cut=14_infl1.5'],
    
]

def get_activities(key_groups):
    activity_results = {}

    for keys in key_groups:
        for k in keys:
            print(k,'JE SUIS LE PATH',f"{root_dir}{scale_expes[k][0]}ema_scale.npy")
            betas = np.load(f"{root_dir}{scale_expes[k][0]}ema_scale.npy")[-1000:].max(axis=0)
            stds = np.load(f"{root_dir}{scale_expes[k][0]}ema_scale.npy")[-1000:].std(axis=0)
            print(k, betas.shape)
            if ((stds/betas) > 0.03).any():
                print("dev beta",stds/betas)
            activity_results[k] = betas.mean()

    return activity_results

def get_loss(key_groups):
    loss_results = {}
    loss_std = {}
    for keys in key_groups:
        for k in keys:
            lo = np.load(f"{root_dir}{scale_expes[k][0]}ema_loss.npy")[-1000:].min()
            stds = np.load(f"{root_dir}{scale_expes[k][0]}ema_loss.npy")[-1000:].std()
            #lostd = np.load(f"{root_dir}{scale_expes[k][0]}ema_loss.npy")[-1000:].std()
            print(k, lo.shape)
            if (stds/lo > 0.03).any():
                print("dev loss",stds/lo)
            loss_results[k] = lo
            #loss_std[k] = lostd
    return loss_results

def get_bias(key_groups):
    loss_results = {}

    for keys in key_groups:
        for k in keys:
            lo = np.load(f"{root_dir}{scale_expes[k][0]}ema_bias.npy")[-1000:].mean()
            print(k, lo.shape)
            loss_results[k] = lo
    return loss_results

def plot_activity_loss(act_data,loss_data, add_name=""):
    fig, ax = plt.subplots(figsize=(8,6))
    for keys in key_groups:
        loss = []
        activities = []
        for k in keys:
            loss.append(loss_data[k])
            activities.append(act_data[k])
        loss = np.array(loss)
        activities = np.array(activities)
        ax.plot(activities, loss, color=colors[attributes[k]['cut']], marker="d", label = str(attributes[k]['cut']))
        #ax.scatter(activities, ff_crps, color=colors[attributes[k]['cut']], marker="+", label = str(attributes[k]['cut']))
        for i, txt in enumerate([attributes[k]['inflt'] for k in keys]):
            ax.annotate((txt), (activities[i], loss[i]),size=18)
    
    plt.title(f"{add_name}"),
    plt.xlabel("Mean(betas)", fontsize=18)
    plt.ylabel(f"Average loss", fontsize=18)
    plt.tick_params(direction='in', length=12, width=1)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.savefig(f"activities_loss_{add_name}.pdf")
    plt.close()
    

if __name__=="__main__":
    
    activities = get_activities(key_groups)
    losses = get_loss(key_groups)
    bias = get_bias(key_groups)
    plot_activity_loss(activities, losses, add_name='Sigmoid stdmean rruvt')
    #plot_activity_loss(activities, bias, add_name='bias')