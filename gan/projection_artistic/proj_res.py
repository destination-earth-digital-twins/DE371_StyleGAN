
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/moldovang/sxbigdata1/stylegan2-pytorch-master')

from utils import (

    real_fake,

    )


import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm# Collect all the font names available to matplotlib

#import plotly.figure_factory as ff

import matplotlib.gridspec as gridspec
import torch
from torch.nn import functional as F
import matplotlib.gridspec as gridspec

font_names = [f.name for f in fm.fontManager.ttflist]
#print(font_names)

mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2
#plt.rcParams['text.usetex'] = True




#print("sizes are", real.size(), fake.size())
# img_grid_real = torchvision.utils.make_grid(real[:8,:1,:,:],)
# img_grid_fake = torchvision.utils.make_grid(fake[:8,:1,:,:],)
# writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
# writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)







mse_SG = np.load('L_mse_' + str(0) + '_SG' + '.npy' )
mse_standard = np.load('L_mse_' + str(0) + '_standard' + '.npy' )
mse_Wplus = np.load('L_mse_' + str(0) + '_Wplus' + '.npy' )
avg_var = np.load('Avg_var_' + str(0) + '_SG' + '.npy' )
#mse_sgan = np.load('STYLE_GAN.npy')    
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes([0, 0, 1, 1])
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)

# Edit the major and minor ticks of the x and y axes
ax.xaxis.set_tick_params(which='major', size=8, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=8, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

ax.plot(mse_SG, linewidth=1, color='black', label='StyleGAN')
ax.plot(mse_standard, linewidth=1, color='blue', label='AROME W')
ax.plot(mse_Wplus, linewidth=1, color='grey', label='AROME W+')
#ax.plot(mse_sgan, linewidth=1, color='blue', label='styleGAN2')
ax.set_yscale('log')

ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=14)
plt.savefig("mse_err"+".pdf", bbox_inches='tight')


###########################################################################################

Ens_proj_var_wplus = np.load('Sample_fake_0_wplus.npy' )
Ens_proj_var_w = np.load('Sample_fake_0_w.npy' )
Ens_real_var = np.load('Sample_real_0.npy' )

Ens_proj_var_wplus = np.squeeze(Ens_proj_var_wplus, axis=0)
Ens_proj_var_w = np.squeeze(Ens_proj_var_w, axis=0)
Ens_real_var = np.squeeze(Ens_real_var, axis=0)

fig, axs = plt.subplots(3, 3, figsize=(6, 4.9))
    
gs1 = gridspec.GridSpec(3, 3)
gs1.update(wspace=0.175, hspace=0.175) # set the spacing between axes. 

#plt.rcParams["figure.figsize"] = [50.00, 3.50]
plt.rcParams["figure.autolayout"] = True

axs = plt.subplot(gs1[0])
c = axs.pcolor(Ens_proj_var_w[2,:,:], cmap="coolwarm",vmin = Ens_real_var[2,:,:].min(), vmax = Ens_real_var[2,:,:].max())
axs.set_title('t2m')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')   
cb.ax.tick_params(labelsize=8)    
cb.ax.yaxis.get_offset_text().set(size=8)    
cb.outline.set_linewidth(1)              
cb.update_ticks()

axs = plt.subplot(gs1[3])
c = axs.pcolor(Ens_proj_var_wplus[2,:,:], cmap="coolwarm",vmin = Ens_real_var[2,:,:].min(), vmax = Ens_real_var[2,:,:].max())
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')   
cb.ax.tick_params(labelsize=8)    
cb.ax.yaxis.get_offset_text().set(size=8)    
cb.outline.set_linewidth(1)              
cb.update_ticks()

axs = plt.subplot(gs1[6])
c = axs.pcolor(Ens_real_var[2,:,:], cmap="coolwarm", vmin = Ens_real_var[2,:,:].min(), vmax = Ens_real_var[2,:,:].max())
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                           
cb.update_ticks()

axs = plt.subplot(gs1[1])
c = axs.pcolor(Ens_proj_var_w[0,:,:], cmap="viridis", vmin = Ens_real_var[0,:,:].min(), vmax = Ens_real_var[0,:,:].max())
axs.axis('off')
axs.set_title('u')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right') 
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                         
cb.update_ticks()

axs = plt.subplot(gs1[4])
c = axs.pcolor(Ens_proj_var_wplus[0,:,:], cmap="viridis", vmin = Ens_real_var[0,:,:].min(), vmax = Ens_real_var[0,:,:].max())
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right') 
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                         
cb.update_ticks()

axs = plt.subplot(gs1[7])
c = axs.pcolor(Ens_real_var[0,:,:], cmap="viridis", vmin = Ens_real_var[0,:,:].min(), vmax = Ens_real_var[0,:,:].max())
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                          
cb.update_ticks()

axs = plt.subplot(gs1[2])
c = axs.pcolor(Ens_proj_var_w[1,:,:], cmap="viridis", vmin = Ens_real_var[1,:,:].min(), vmax = Ens_real_var[1,:,:].max())
axs.axis('off')
axs.set_title('v')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                          
cb.update_ticks()

axs = plt.subplot(gs1[5])
c = axs.pcolor(Ens_proj_var_wplus[1,:,:], cmap="viridis", vmin = Ens_real_var[1,:,:].min(), vmax = Ens_real_var[1,:,:].max())
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                          
cb.update_ticks()

axs = plt.subplot(gs1[8])
c = axs.pcolor(Ens_real_var[1,:,:], cmap="viridis", vmin = Ens_real_var[1,:,:].min(), vmax = Ens_real_var[1,:,:].max())
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                          
cb.update_ticks()


plt.savefig("comparison_real_proj.jpg",  dpi=600, transparent=False, bbox_inches='tight')

#################################################################################################

fig, axs = plt.subplots(2, 3, figsize=(6, 3.15))

#plt.rcParams["figure.figsize"] = [50.00, 3.50]
plt.rcParams["figure.autolayout"] = True
vmin = 0.00001
vmax = 0.03
gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.175, hspace=0.175) # set the spacing between axes. 



Dv_w=abs(Ens_real_var[1,:,:]-Ens_proj_var_w[1,:,:])
Du_w=abs(Ens_real_var[0,:,:]-Ens_proj_var_w[0,:,:])
Dt_w=abs(Ens_real_var[2,:,:]-Ens_proj_var_w[2,:,:])

Dv_wplus=abs(Ens_real_var[1,:,:]-Ens_proj_var_wplus[1,:,:])
Du_wplus=abs(Ens_real_var[0,:,:]-Ens_proj_var_wplus[0,:,:])
Dt_wplus=abs(Ens_real_var[2,:,:]-Ens_proj_var_wplus[2,:,:])

axs = plt.subplot(gs1[0])
c=axs.pcolor(Dt_w, cmap="Greys", vmax = vmax/1., vmin = vmin)
axs.set_title('t2m')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[1])
c=axs.pcolor(Du_w, cmap="Greys", vmax = vmax, vmin = vmin)
axs.set_title('u')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[2])
c=axs.pcolor(Dv_w, cmap="Greys", vmax = vmax, vmin = vmin)
axs.set_title('v')
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[3])
c=axs.pcolor(Dt_wplus, cmap="Greys", vmax = vmax/1., vmin = vmin)
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[4])
c=axs.pcolor(Du_wplus, cmap="Greys", vmax = vmax, vmin = vmin)
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()

axs = plt.subplot(gs1[5])
c=axs.pcolor(Dv_wplus, cmap="Greys", vmax = vmax, vmin = vmin)
axs.axis('off')
cb = fig.colorbar(c,ax=axs)     
cb.formatter.set_powerlimits((0, 0))
cb.ax.yaxis.set_offset_position('right')
cb.ax.tick_params(labelsize=8)
cb.ax.yaxis.get_offset_text().set(size=8)
cb.outline.set_linewidth(1)                                 
cb.update_ticks()


plt.savefig("compa_AE_proj.jpg",  dpi=600, transparent=False, bbox_inches='tight')





