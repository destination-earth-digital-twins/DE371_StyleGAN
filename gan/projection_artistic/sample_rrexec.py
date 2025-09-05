import numpy as np
import artistic as art
import argparse

# TODO : Not sure we need to keep that file

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    parser.add_argument('--fake_data_dir', type = str,  default ='')
    parser.add_argument('--Path_out', type = str,  default ='')
    parser.add_argument('--data_dir', type = str, default='/project/home/p200177/DE_371/datasets/dataset_Meteo_France/IS_1_1.0_0_0_0_0_0_256_large_lt_done/')
    parser.add_argument('--single_sample', type = str, default='')
    params = parser.parse_args()
    

    CI = (78, 206, 55, 183)
    CI = (0, 256, 0, 256)
    Maxs = 1 #np.load(data_dir+'max_log_rr_imp.npy')[0:1].reshape(1, 1, 1)
    Means = 1 #np.load(data_dir+'mean_log_rr_imp.npy')[0:1].reshape(1, 1, 1)

    #var_names = [('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]

    #var_names = [('rr', 'mm')]
    var_names = [('t2m', 'K')]
    index = 0
    lat_dim = 512

    n_plots = 4

    #var_names=[('v', 'm/s')]

    # data_f = [1016,274,721] # 274 150
    #data_f = [184, 771, 62]

    # data_f=[916,449,733]
    # data_f = [1016, 228, 721] #(8 3 58500)
    #data_f = random.sample(range(128),n_plots-1)

    #Data_fake = np.load(fake_data_dir + '_Fsample_{}_0.npy'.format(step))

    # print(data_f)

    #data_flist=[Data_fake[data_f[j]] for j in range(n_plots-1)]

    data_flist = np.expand_dims(np.load(single_sample),axis=0)

    data_flist = np.concatenate([np.expand_dims(data_flist[:,:,:,i], axis=0) for i in range(4)],axis=0)
    #del Data_fake


    """litrue = pd.read_csv(data_dir + 'labels/-0-2_50-0_1-0_0-0_20-0_0-0/INST1.csv')['Name'].to_list()

    data_r_name = '_sample2473'#random.sample(litrue,1)[0]

    #data_r_name = '_sample19117.npy'
    
    print(data_r_name)

    Data_real = np.load(data_dir + data_r_name + '.npy').astype(np.float32)

    #Ens_proj_var_wplus = np.load(Path_samples+f'Fsemble_{lat_dim}_3.0_3.0.npy')
    #Ens_proj_var_w = np.load('Sample_fake_0_w.npy' )
    #Ens_real_var = np.load(Path_samples+f'Rsemble_{lat_dim}_3.0_3.0.npy')

    #Ens_proj_var_wplus = Ens_proj_var_wplus
    #Ens_proj_var_w = Ens_proj_var_w
    #Ens_real_var = Ens_real_var

    
    channels = 1  # only 1 variable plot
    #data = np.zeros((n_plots, channels, 128, 128))

    data = np.zeros((n_plots, channels, 256, 256))

    data[0] = Data_real[0:1]
    #data0 = data0[0,0,:,:]
    #data0 = np.expand_dims(data0, axis=0)

    #Data_fake = Data_fake[0,:,:]
    # Data_fake = np.expand_dims(Data_fake, axis=0)
    # Data_fake = np.expand_dims(Data_fake, axis=0)

    # print(data0.shape)

    #data0 = art.standardize_samples(Ens_real_var, normalize=[0], norm_vectors=(Means, Maxs),
    #                                chan_ind=[0, 1, 2], ref_chan_ind=[0, 1, 2])[index]
    #data1 = art.standardize_samples(Ens_proj_var_w, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data2 = art.standardize_samples(Ens_proj_var_wplus, normalize=[0], norm_vectors=(Means, Maxs),
    #                                chan_ind=[0, 1, 2], ref_chan_ind=[0, 1, 2])[index]

    #data[0] = art.standardize_samples(Ens_real_var, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[1] = art.standardize_samples(Ens_proj_var_w, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[2] = art.standardize_samples(Ens_proj_var_wplus, normalize=[0], norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])

    #data[0] = abs(data2 - data0)  # data1-data0
    #data[1] = abs(data2 - data0)

    #data[0] = art.standardize_samples(Data_noise_var,norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[0] = art.standardize_samples(Data_noise_var_4,norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])
    #data[1] = art.standardize_samples(Data_noise_var_32,norm_vectors=(Means,Maxs), chan_ind =[0, 1, 2], ref_chan_ind =[0, 1, 2])"""

    can = art.canvasHolder("SE_for_GAN_terrestrial", 256, 256)


    """for j in range(n_plots):#-1):

        data[j] = art.standardize_samples(data_flist[j], normalize=[0],norm_vectors=(Means,Maxs), chan_ind =[0], ref_chan_ind =[0])"""

    # exponentiating
    """for j in range(n_plots-1):

        data[j+1]  = np.exp(data[j+1]) - 1.0"""

    #data[j] = art.standardize_samples(np.expand_dims(Data_interp[j], axis=0), normalize=[0],norm_vectors=(Means,Maxs), chan_ind =[0], ref_chan_ind =[0])
    data = data_flist
    Datamax = data.max(axis=(0, -2, -1))
    #Datamax[0] = Datamax[0]*0.99
    # Datamax[1]=Datamax[1]*0.85
    # Datamax[2]=Datamax[2]*0.99

    Datamin = data.min(axis=(0, -2, -1))

    Datamean = data[1:].mean(axis=(0,-2,-1))
    print(Datamin, Datamax, Datamean)
    print("data shape is", data.shape)

    can.plot_data_normal(data, var_names, Path_out, f'artistic_{lat_dim}_index_{index}.jpg', contrast=False,
                         cvalues=(Datamin, Datamax))
    #can.plot_data_wind(data[:,0:2,:,:], path_plot_full,'new_artistic_wind_63_357000_cbr.png',withQuiver=False)
    #can.plot_data_wind(data[:,0:2,:,:], path_plot_full,'new_artistic_wind_quiver_63_357000_cbr.png',withQuiver=True)

    # can.plot_data_normal(data,var_names,path_plot, 'new_artistic0_progan.png', contrast=True,
    #                     cvalues=(Datamin, Datamax))
