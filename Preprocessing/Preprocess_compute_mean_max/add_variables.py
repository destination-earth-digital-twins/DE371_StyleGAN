import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os
from time import perf_counter


# launch this on sxrecyf -> way too long otherwise

def load_concat(indexes, df, new_im, data_path, save_path):
    df_tmp = df.iloc[indexes]
    name = df_tmp.loc["Name"]
    if not os.path.isfile(save_path + name + '.npy'):
        img = np.load(data_path + name + '.npy')[0:5,:,:].reshape(5,256,256)
        img = np.concatenate((img, new_im[:,:,:,df_tmp["LeadTime"],df_tmp["Member"]].reshape(3,256,256)))
        np.save(save_path + name + '.npy', img)

def add_var(var_names=['z500','t850','tpw850'], csv_path='', data_path='', new_data_path='', save_path=''):
    df = pd.read_csv(csv_path)
    print(df.head())
    t_s = perf_counter()
    for idx, date in enumerate(np.unique(df["Date"].values)):
        if idx%1==0:
            print(idx, perf_counter()-t_s)
            t_s = perf_counter()
        if os.path.isfile(new_data_path+date+'_'+ var_names[0] +'.npy'):
            new_im = np.concatenate([np.load(new_data_path+date+'_'+ var +'.npy')[120:120+256,540:540+256].reshape(1,256,256,8,16)\
                                 for var in var_names], axis=0)
            indexes = df[df["Date"]==date].index
            print("entering pool")
            with Pool(1) as p:
                p.map(partial(load_concat, df=df, new_im=new_im, data_path=data_path, save_path=save_path), indexes)
        
    return 0


if __name__=="__main__":
    csv_path = ''
    var_names = ['z500','t850','tpw850']
    data_path = ""
    new_data_path = ""
    save_path = ""
    add_var(var_names=var_names, csv_path=csv_path, data_path=data_path, new_data_path=new_data_path, save_path=save_path)

