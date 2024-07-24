import pandas as pd 
import numpy as np
import random
import torch
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
        Shape=(3,256,256),
        var_indices=[0,1,2],
        normalization="meanmax",
        Means=None,
        Mins=None,
        Maxs=None
        ):

    df0 = dataframe[(dataframe['Date']==date) & (dataframe['LeadTime']==lt)]

    Nb = len(df0)

    batch = np.zeros((Nb,) + tuple(Shape))
    print(batch.shape)
    for i,s in enumerate(df0['Name']):
        print(i, s)
        sn = np.load(f'{data_dir}{s}.npy')[var_indices,:,:].astype(np.float32)

        batch[i] = sn
    
    # normalise samples and save in pack dir. obs! make sure normalization is done correctly (according to how model was trained)
    if normalization=="meanmax":
        batch = torch.tensor(0.95*(batch - Means) / (Maxs), dtype = torch.float32)
    elif normalization=="minmax":
        batch = torch.tensor(-1. + 2*(batch - Mins) / (Maxs-Mins), dtype = torch.float32)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return batch


def load_batch_sequence_from_date(
        dataframe, 
        date, 
        data_dir, 
        concatenate_variable_and_time=False, 
        dt=3, 
        Shape=(3,256,256), 
        var_indices=[0,1,2],
        normalization="meanmax",
        Means=None,
        Mins=None,
        Maxs=None
    ):
    r''' 

    '''
    batch_sequence=[]
    for member_id in range(16) :
        df0 = dataframe[(dataframe['Date']==date) & (dataframe['Member']==member_id)]
        Nb = len(df0) # nb total leadtime

        sample = np.zeros((Nb//dt,) + tuple(Shape))

        for i,s in enumerate(df0['Name']):
            if i%dt == 0:
                sn = np.load(f'{data_dir}{s}.npy')[var_indices,:,:].astype(np.float32)
                # normalization
                if normalization=="meanmax":
                    sn = np.array(0.95*(sn - Means) / (Maxs), dtype = np.float32)
                elif normalization=="minmax":
                    sn = np.array(-1. + 2*(sn - Mins) / (Maxs-Mins), dtype = torch.float32)
                else:
                    raise ValueError(f"Unknown normalization: {normalization}")
                sample[i//dt] = sn
        if concatenate_variable_and_time :
            lt, var, x, y = np.shape(sample)
            sample = sample.reshape((lt*var,x,y))

        batch_sequence.append(sample)

    return torch.tensor(batch_sequence, dtype=torch.float32)

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

def collate_w_ensemble(data_dir, members, lead_time , var_indices):
    """
    Fetch individual members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    nb_members = len(members)

    batch = np.load(data_dir + f'w_ge_{lead_time}_875.npy').astype(np.float32)[members]

    print(batch.shape)

    return batch

def collate_R_ensemble(data_dir, members, lead_time , var_indices):
    """
    Fetch individual members of the same forecast at a given lead time (as isolated files) 
    and feed them as one single array
    """

    nb_members = len(members)

    batch = np.zeros((nb_members,3,256,256), dtype = np.float32)

    dataloaded = np.load(data_dir + f'Rsemble_{lead_time}_875.npy', mmap_mode='r').astype(np.float32)
    
    batch = dataloaded[members]

    return batch


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
