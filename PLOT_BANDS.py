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
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['font.size'] = 20  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['figure.figsize'] = [8.,6]
mpl.rcParams['text.usetex'] = True
#m = 1

RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 

ii = 14                                                                        # cell index
nn = 2*(ii**2+(ii+1)*ii+(ii+1)**2)                                                    # number of atoms
lconst = 3.161                                                                  # lattice constant
dis = 3.1                                                                      # cell index
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

num_GK = 50                                                                    # number of k-point per hgh symmetry line
num_KM = 28                                                                    # number of k-point per hgh symmetry line                                                                # interlayer distance
num_MG = 50   

# =============================================================================
# file_BANDS = open('mu.dat','r')
# mu = np.loadtxt(file_BANDS)
# file_BANDS.close()
# =============================================================================

file = open('../bands.dat','r')
MAT_BANDS = np.loadtxt(file)
file.close()

E_MAX = np.amax(MAT_BANDS)

# =============================================================================
# file = open('EQ_BC_LOOP_PATH.dat','r')
# BC_LOOP = np.loadtxt(file)
# file.close()
# =============================================================================
  
fig1 = plt.figure(1)
gs1 = gridspec.GridSpec(1, 1)
ax11 = fig1.add_subplot(gs1[0,0])

ax11.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
ax11.set_xticks([0, num_GK, num_GK+num_KM, num_GK+num_KM+num_MG])
ax11.set_xticklabels([r'$\mathrm{\Gamma}$',  r'$\mathrm{K}$', r'$\mathrm{M}$', r'$\mathrm{\Gamma}$'])
#ax11.set_xticks([0, num_GK, num_GK+num_KM/2, num_GK+num_KM, num_GK+num_KM+num_MG])
#ax11.set_xticklabels([r'$\mathrm{\Gamma}$',  r'$\mathrm{K1}$', r'$\mathrm{M}$', r'$\mathrm{K1}$', r'$\mathrm{\Gamma}$'])
ax11.plot(MAT_BANDS[:,1250:], 'k', linewidth=2.0)
ax11.set_ylim(E_MAX-0.10,E_MAX+0.02)

plt.tight_layout()
plt.show()

#print(np.shape(BC_LOOP))

#BC_BANDS = np.sum(BC_LOOP,axis=0)

#print(np.shape(BC_BANDS))
                                                      # slope of super cell diagonal

# superlattice bravis translational vectors
lvec = np.zeros((2,2)) 
lvec[0,0] =  79.4034118652
lvec[0,1] =  0.0
lvec[1,0] =  39.7017059326
lvec[1,1] =  68.7653718225

# Real space basis vectors of super cell
A1 = np.array([lvec[0,0], lvec[0,1], 0.])                                      # (lconst*AA)
A2 = np.array([lvec[1,0], lvec[1,1], 0.])                                      # (lconst*AA)
A3 = np.array([0.0, 0.0, 1.0])*(dis/lconst)                                    # (lconst*AA)

# Reciprocal space basis vectors of super cell
B1 = 2.*np.pi*np.cross(A2,A3)/(lconst*np.dot(A1,np.cross(A2,A3)))                       # 1/(lconst*AA)
B2 = 2.*np.pi*np.cross(A3,A1)/(lconst*np.dot(A2,np.cross(A3,A1)))                       # 1/(lconst*AA)
B3 = 2.*np.pi*np.cross(A1,A2)/(lconst*np.dot(A3,np.cross(A1,A2)))                       # 1/(lconst*AA)

AREA_BZ=np.linalg.norm(np.cross(B1,B2))/(64*64)
#print(np.amax(BC_LOOP[:,328]))

# =============================================================================
# fig1 = plt.figure(2)
# gs1 = gridspec.GridSpec(1, 1)
# ax = fig1.add_subplot(gs1[0,0])
# ax.plot(BC_BANDS*AREA_BZ/(2*np.pi))
# ax.set_xlim(320,330)
# ax.set_ylim(-10,10)
# plt.plot()
# =============================================================================
