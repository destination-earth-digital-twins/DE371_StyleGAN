import pandas as pd 
import numpy as np
import random
import torch
import os
import copy
#random.seed(0)

def str2intlist(li):
    if type(li)==list:
        li2 = [int(p) for p in li]
        return li2
    
    elif type(li)==str:
        li2 = li[1:-1].split(',')
        li3 = [int(p) for p in li2]
        return li3

    else : 
        raise ValueError("li argument must be a string or a list, not '{}'".format(type(li)))



def load_batch_from_timestamp(
        dataframe,
        date,
        lt,
        data_dir,
        Shape,
        var_indices,
        normalization,
        Means=None,
        Mins=None,
        Maxs=None, 
        apply_log_transform = False 
        ):

    df0 = dataframe[(dataframe['Date']==date) & (dataframe['LeadTime']==lt)]

    Nb = len(df0)

    batch = np.zeros((Nb,) + tuple(Shape))

    for i,s in enumerate(df0['Name']):
        sn = np.load(f'{data_dir}{s}.npy')[var_indices,:,:].astype(np.float32)

        batch[i] = sn
    
    # normalise samples and save in pack dir. obs! make sure normalization is done correctly (according to how model was trained)
    batch = torch.tensor(batch, dtype=torch.float32)
    norm_batch = normalize(
                        data=batch,
                        normalization_type=normalization,
                        Means=Means,
                        Mins=Mins,
                        Maxs=Maxs,
                        apply_log_transform=apply_log_transform
                        )


    return batch, norm_batch


def load_batch_sequence_from_date(
        dataframe, 
        date, 
        data_dir, 
        concatenate_variable_and_time=False, 
        dt=3, 
        Shape=(3,256,256), 
        var_indices=[1,2,3],
        normalization="meanmax",
        Means=None,
        Mins=None,
        Maxs=None,
        apply_log_transform=False
    ):
    r''' 

    '''
    batch_sequence=[]
    for member_id in range(16) :
        df0 = dataframe[(dataframe['Date']==date) & (dataframe['Member']==member_id)]
        Nb = len(df0) # nb total leadtime
        if dt == 6:
            sample = np.zeros((8,) + tuple(Shape))
        else :
            sample = np.zeros((Nb//dt,) + tuple(Shape))


        for i,s in enumerate(df0['Name']):
            if i%dt == 0:
                sn = np.load(f'{data_dir}{s}.npy')[var_indices,:,:].astype(np.float32)
                # normalization
                sn_norm = normalize(
                                data=sn,
                                normalization_type=normalization,
                                Means=Means,
                                Mins=Mins,
                                Maxs=Maxs,
                                apply_log_transform=apply_log_transform
                        )
                sample[i//dt] = sn_norm
        if concatenate_variable_and_time :
            lt, var, x, y = np.shape(sample)
            sample = sample.reshape((lt*var,x,y))

        batch_sequence.append(sample)

    return torch.tensor(np.array(batch_sequence), dtype=torch.float32)

def rescale(generated, Mean, Max, scale) : 
    r''' 
    Rescale generated data w.r.t. given mean, max and scale
    '''
    return scale * Max * generated + Mean

def collate_ensemble(data_dir, start_member, stop_member, lead_time , var_indices):
    r"""
    Fetch individual members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    nb_members = stop_member - start_member + 1

    batch = np.zeros((nb_members,3,256,256), dtype = np.float32)

    for i, mb in zip(range(nb_members), range(start_member, stop_member + 1)):
    
        batch[i] = np.load(data_dir + f'_grand_sample_{lead_time}_875.npy').astype(np.float32)[mb,var_indices]

    return batch

def collate_w_ensemble(data_dir, members, lead_time , var_indices, inv_step):
    """
    Fetch individual members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    nb_members = len(members)
    if os.path.exists(data_dir + f'w_ge_{lead_time}_875.npy') :
        batch = np.load(data_dir + f'w_ge_{lead_time}_875.npy').astype(np.float32)[members]
    elif os.path.exists(data_dir + f'w_0_15_{lead_time}_{inv_step}.npy') :
        batch=[]
        for i in range(0, 865, 16):
            if i == 864:
                batch.append(np.load(data_dir + f'w_{i}_{i+10}_{lead_time}_{inv_step}.npy', mmap_mode='r').astype(np.float32)) 
            else :   
                batch.append(np.load(data_dir + f'w_{i}_{i+15}_{lead_time}_{inv_step}.npy', mmap_mode='r').astype(np.float32))
        batch = np.vstack(batch).astype(np.float32)

    print(batch.shape)

    return batch

def collate_R_ensemble(data_dir, members, lead_time , var_indices, all_data = False):
    """
    Fetch individual members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    if os.path.exists(data_dir + f'Rsemble_{lead_time}_875.npy') :
        dataloaded = np.load(data_dir + f'Rsemble_{lead_time}_875.npy', mmap_mode='r').astype(np.float32)
    elif os.path.exists(data_dir + f'Rsemble_0_15_{lead_time}.npy') :
        dataloaded=[]
        for i in range(0, 865, 16):
            if i == 864:
                dataloaded.append(np.load(data_dir + f'Rsemble_{i}_{i+10}_{lead_time}.npy', mmap_mode='r').astype(np.float32)) 
            else :   
                dataloaded.append(np.load(data_dir + f'Rsemble_{i}_{i+15}_{lead_time}.npy', mmap_mode='r').astype(np.float32))
        dataloaded = np.vstack(dataloaded).astype(np.float32)
    else :
        raise FileNotFoundError

    if not all_data:
        return dataloaded[members]
    else :
        return dataloaded
    

def collate_inv_ensemble(data_dir, members, lead_time , var_indices, inv_step, all_data=False):
    """
    Fetch individual inverted members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    if os.path.exists(data_dir + f'invertFsemble_{lead_time}_875.npy') :
        dataloaded = np.load(data_dir + f'invertFsemble_{lead_time}_875.npy', mmap_mode='r').astype(np.float32)
    elif os.path.exists(data_dir + f'invertFsemble_0_15_{lead_time}_{inv_step}.npy') :
        dataloaded=[]
        for i in range(0, 865, 16):
            if i == 864:
                dataloaded.append(np.load(data_dir + f'invertFsemble_{i}_{i+10}_{lead_time}_{inv_step}.npy', mmap_mode='r').astype(np.float32)) 
            else :   
                dataloaded.append(np.load(data_dir + f'invertFsemble_{i}_{i+15}_{lead_time}_{inv_step}.npy', mmap_mode='r').astype(np.float32))
        dataloaded = np.vstack(dataloaded).astype(np.float32)
    else :
        raise FileNotFoundError

    if not all_data:
        return dataloaded[members]
    else :
        return dataloaded


def collate_gen_ensemble(data_dir, members, lead_time , var_indices, inv_step, all_data=False):
    """
    Fetch individual generated members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    if os.path.exists(data_dir + f'genFsemble_{lead_time}_875.npy') :
        dataloaded = np.load(data_dir + f'genFsemble_{lead_time}_875.npy', mmap_mode='r').astype(np.float32)
    elif os.path.exists(data_dir + f'genFsemble_0_{lead_time}_{inv_step}.npy') :
        dataloaded=[]
        for i in range(0, 50):
            dataloaded.append(np.load(data_dir + f'genFsemble_{i}_{lead_time}_{inv_step}.npy', mmap_mode='r').astype(np.float32)) 
        dataloaded = np.vstack(dataloaded).astype(np.float32)
    else :
        raise FileNotFoundError

    if not all_data:
        return dataloaded[members]
    else :
        return dataloaded

def correct_lt(lt):
    if lt<=24:
        lt_corr = (lt - 3) // 3
    else: 
        lt_corr = lt
    return lt_corr


lstlbc = [2,20,9,5,32,15,19,21,13,1,34,12,10,31,23,11,8,24,29,22,28,25,6,33,14,7,30,27,0,18,4,26,3,16,17]
lstic  = list(range(1,26))
Ns = 16
Nlbc = 35 
def initsmall():
    """
    
    Select distinct boundary and initial conditions for AROME-EPS members
    Func  by L. Raynaud
    
    """
    yic = random.sample(lstic, Ns)
    ybc = random.sample(lstlbc, Ns)
    mb = np.zeros((Ns))
    # Find members corresponding to yic/ybc pairs
    for k in range(Ns):
        loc_bc = np.where(np.asarray(lstlbc)==ybc[k])
        #index member of the PEARO experiment start from 1
        #if python storage of members start at 0 remove '+1'
        mb[k] = ( yic[k] - 1 ) * Nlbc + loc_bc[0][0] # + 1
    return mb


def normalize(data, normalization_type, Means=None, Mins=None, Maxs=None, apply_log_transform=True):
    """
    Normalizes the data and if necessary does the log-transform.

        Args:
            data (torch.Tensor): The normalized data.
            normalization_type (str): Type of normalisation used ('meanmax', 'minmax' or '').
            Means (torch.Tensor, optional): Means used for normalisation (if applicable).
            Mins (torch.Tensor, optional): Minima used for normalisation (if applicable).
            Maxs (torch.Tensor, optional): Maxima used for normalisation (if applicable).
            apply_log_transform (bool): If True, also reverses the log transformation.

        Returns:
            torch.Tensor: The denormalized data.
    """
    
    normalized_data = copy.copy(data)

    if apply_log_transform:
        normalized_data[:,0,:,:]=torch.log(1+normalized_data[:,0,:,:])   

    if normalization_type == "meanmax":
        if Means is None or Maxs is None:
            raise ValueError("Means et Maxs must be supplied to denormalise with 'meanmax'.")
        normalized_data = torch.tensor(0.95*(normalized_data - Means) / (Maxs), dtype = torch.float32)
    elif normalization_type == "minmax":
        if Mins is None or Maxs is None:
            raise ValueError("Mins et Maxs must be supplied to denormalise with 'minmax'.")
        normalized_data = torch.tensor(-1. + 2*(normalized_data - Mins) / (Maxs-Mins), dtype = torch.float32)
    else:
        raise ValueError(f"Type de normalisation inconnu: {normalization_type}")
      

    return normalized_data

def denormalize(data, normalization_type, Means=None, Mins=None, Maxs=None, apply_log_transform=True):
    """
    Denormalizes the data by inverting the normalization transforms and, if necessary, the log-transform.

        Args:
            data (torch.Tensor): The normalized data.
            normalization_type (str): Type of normalisation used ('meanmax', 'minmax' or '').
            Means (torch.Tensor, optional): Means used for normalisation (if applicable).
            Mins (torch.Tensor, optional): Minima used for normalisation (if applicable).
            Maxs (torch.Tensor, optional): Maxima used for normalisation (if applicable).
            apply_log_transform (bool): If True, also reverses the log transformation.

        Returns:
            torch.Tensor: The denormalized data.
    """
    #Inverser la normalisation
    if normalization_type == "meanmax":
        if Means is None or Maxs is None:
            raise ValueError("Means et Maxs must be supplied to denormalise with 'meanmax'.")
        denormalized_data = (data * Maxs / 0.95) + Means
    elif normalization_type == "minmax":
        if Mins is None or Maxs is None:
            raise ValueError("Mins et Maxs must be supplied to denormalise with 'minmax'.")
        denormalized_data = ((data + 1) * (Maxs - Mins) / 2) + Mins
    elif normalization_type == "":
        denormalized_data = data  
    else:
        raise ValueError(f"Type de normalisation inconnu: {normalization_type}")
    # Reverse the logarithmic transformation
    
    if apply_log_transform:
        denormalized_data[:,0,:,:] = torch.exp(denormalized_data[:,0,:,:]) - 1

    return denormalized_data