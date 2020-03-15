# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:00:25 2018

@author: toppgabr
"""

import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from matplotlib.colors import ListedColormap
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['font.size'] = 16  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['figure.figsize'] = [16.,6]
mpl.rcParams['text.usetex'] = True

#m = 1

m_max = 1
ii = 14                                                                        # cell index
nn = 2*(ii**2+(ii+1)*ii+(ii+1)**2)                                               # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

num_GK = 50                                                                    # number of k-point per hgh symmetry line
num_KM = 28                                                                    # number of k-point per hgh symmetry line                                                                # interlayer distance
num_MG = 50   

RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 
GREY = '#bdbdbd'

case = 10
#------------------------------------------------------------------------------DATA
file_BANDS = open('../bands.dat','r')
MAT_BANDS = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('DATA/bands_floquet_'+str(case)+'.dat','r')
MAT_BANDS_FLOQ = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('DATA/overlap_floquet_'+str(case)+'.dat','r')
MAT_OVERLAP = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('DATA/overlap_floquet_s_'+str(case)+'.dat','r')
MAT_OVERLAP_s = np.loadtxt(file_BANDS)
file_BANDS.close()

file_BANDS = open('DATA/overlap_floquet_p_'+str(case)+'.dat','r')
MAT_OVERLAP_p = np.loadtxt(file_BANDS)
file_BANDS.close()

MAT_OVERLAP = MAT_OVERLAP/np.amax(MAT_OVERLAP)
MAT_OVERLAP_s = MAT_OVERLAP_s/np.amax(MAT_OVERLAP_s)
MAT_OVERLAP_p = MAT_OVERLAP_p/np.amax(MAT_OVERLAP_p)

E_MAX = np.amax(MAT_BANDS)

ymin = 1.425
ymax = 1.500

#------------------------------------------------------------------------------PLOTS
fig1 = plt.figure(1)
gs1 = gridspec.GridSpec(1, 2)
#------------------------------------------------------------------------------PLOT00
ax11 = fig1.add_subplot(gs1[0,0])
ax11.set_title(r"$\mathrm{Overlap}$ $\mathrm{s-bands}$")
ax11.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax11.set_xticks([0, num_GK, num_GK+num_KM, num_GK+num_KM+num_MG])
ax11.set_xticklabels([r'$\mathrm{\Gamma}$',  r'$\mathrm{K}$', r'$\mathrm{M}$', r'$\mathrm{\Gamma}$'])
ax11.set_ylim(ymin,ymax)

# Choose colormap   
colormap = plt.cm.get_cmap('Reds', 100)
# Get the colormap colors
my_cmap = colormap(np.arange(colormap.N))
# Set alpha
#my_cmap[:,-1] = np.linspace(0, 1, colormap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)

kk = np.linspace(0,127,128)
nband = (2*m_max+1)*nn-100


for i in range(np.size(MAT_BANDS_FLOQ[0,nband:])):
    size = MAT_OVERLAP[:,nband+i]*40   
    sc1 = ax11.scatter(kk, MAT_BANDS_FLOQ[:,nband+i], c=MAT_OVERLAP_s[:,nband+i], cmap=my_cmap, vmin=0.0, vmax=1.0, linewidth=0.1, s=size)
ax11.plot(MAT_BANDS[:,0], '--', color='k', linewidth=1.0, label=r'$\mathrm{GS}$')  
ax11.plot(MAT_BANDS[:,1250:], '--', color='k', linewidth=1.0)   
ax11.plot(MAT_BANDS_FLOQ[:,nband:], '-', color='k', linewidth=0.5)  
ax11.plot(MAT_BANDS_FLOQ[:,0], '-', color='k', linewidth=0.5, label=r'$\mathrm{Floquet}$')  
    
cb1 = plt.colorbar(sc1)
cb1.set_ticks([0.0, 1.0])
cb1.set_ticklabels(['0.0', '1.0'])
plt.legend(loc="lower right")

#------------------------------------------------------------------------------PLOT01
ax12 = fig1.add_subplot(gs1[0,1])
ax12.set_title(r"$\mathrm{Overlap}$ $\mathrm{p-bands}$")
ax12.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax12.set_xticks([0, num_GK, num_GK+num_KM, num_GK+num_KM+num_MG])
ax12.set_xticklabels([r'$\mathrm{\Gamma}$',  r'$\mathrm{K}$', r'$\mathrm{M}$', r'$\mathrm{\Gamma}$'])
ax12.set_ylim(ymin,ymax)

# Choose colormap   
colormap = plt.cm.get_cmap('Blues', 100)
# Get the colormap colors
my_cmap = colormap(np.arange(colormap.N))
# Set alpha
#my_cmap[:,-1] = np.linspace(0, 1, colormap.N)
# Create new colormap
my_cmap = ListedColormap(my_cmap)

kk = np.linspace(0,127,128)


for i in range(np.size(MAT_BANDS_FLOQ[0,nband:])):
    size = MAT_OVERLAP[:,nband+i]*40   
    sc2 = ax12.scatter(kk, MAT_BANDS_FLOQ[:,nband+i], c=MAT_OVERLAP_p[:,nband+i], cmap=my_cmap, vmin=0.0, vmax=1.0, linewidth=0.1, s=size)
ax12.plot(MAT_BANDS[:,0], '--', color='k', linewidth=1.0, label=r'$\mathrm{GS}$')  
ax12.plot(MAT_BANDS[:,1250:], '--', color='k', linewidth=1.0)  
ax12.plot(MAT_BANDS_FLOQ[:,nband:], '-', color='k', linewidth=0.5)  
ax12.plot(MAT_BANDS_FLOQ[:,0], '-', color='k', linewidth=0.5, label=r'$\mathrm{Floquet}$') 
  
cb2 = plt.colorbar(sc2)
cb2.set_ticks([0.0, 1.0])
cb2.set_ticklabels(['0.0', '1.0'])

plt.legend(loc="lower right")
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.07, left=0.10, right=0.99, hspace=0.0, wspace=0.15)
plt.show()
