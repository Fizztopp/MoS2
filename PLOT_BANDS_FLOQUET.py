# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:15:37 2018

@author: toppgabr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:00:25 2018

@author: toppgabr
"""

import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib as mpl
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['font.size'] = 16  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 22
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['figure.figsize'] = [8.,6]
mpl.rcParams['text.usetex'] = True

m_max = 1
ii = 14                                                                # cell index
nn = 2*(ii**2+(ii+1)*ii+(ii+1)**2)                                            # number of atoms
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

case = 2

for mm in range(1):
    print(mm)
    file_BANDS = open('../bands.dat','r')
    MAT_BANDS = np.loadtxt(file_BANDS)
    file_BANDS.close()

    file_BANDS = open('DATA/bands_floquet_'+str(case)+'.dat','r')
    MAT_BANDS_FLOQ = np.loadtxt(file_BANDS)
    file_BANDS.close()
    
# =============================================================================
#     file_BANDS = open('bands_strob.dat','r')
#     MAT_BANDS_FLOQ_STROB = np.loadtxt(file_BANDS)
#     file_BANDS.close()
# =============================================================================
    
    file_BANDS = open('DATA/overlap_floquet_'+str(case)+'.dat','r')
    MAT_OVERLAP = np.loadtxt(file_BANDS)
    file_BANDS.close()
    
# =============================================================================
#     file_BANDS = open('FLOQUET_BC_LOOP_PATH.dat','r')
#     FLOQUET_BC_LOOP_PATH = np.loadtxt(file_BANDS)
#     file_BANDS.close()
# =============================================================================
    E_MAX = np.amax(MAT_BANDS)
    
    fig1 = plt.figure(2)
    gs1 = gridspec.GridSpec(1, 1)
    
    ax12 = fig1.add_subplot(gs1[0,0])
    ax12.set_ylabel(r'$\mathrm{Energy}$ $\mathrm{(eV)}$')
    ax12.set_xticks([0 , 32, 95,  128])
    ax12.set_xticklabels(['', r'$\mathrm{K1}$',  r'$\mathrm{K2}$', ''])
    
    ax12.plot(MAT_BANDS_FLOQ[:,int((2*m_max+1)*nn-100):], 'b-', linewidth=2.0)
# =============================================================================
#     ax12.plot(MAT_BANDS_FLOQ[:,992], 'r', linewidth=2.0)
#     ax12.plot(MAT_BANDS_FLOQ[:,991], 'r', linewidth=2.0)
#     ax12.plot(MAT_BANDS_FLOQ[:,990], 'r', linewidth=2.0)
#     ax12.plot(MAT_BANDS_FLOQ[:,989], 'r', linewidth=2.0)
#     ax12.plot(MAT_BANDS_FLOQ[:,988], 'r', linewidth=2.0)
#     ax12.plot(MAT_BANDS_FLOQ[:,987], 'r', linewidth=2.0)
# =============================================================================
    ax12.plot(MAT_BANDS[:,1250:], 'k--', linewidth=1.0)
    #ax12.plot(MAT_BANDS[:,0], 'k--', linewidth=1.0, label=r'$\mathrm{GS}$')
    ax12.set_ylim(E_MAX-0.13,E_MAX+0.02)
    ax12.set_xticks([0, num_GK, num_GK+num_KM, num_GK+num_KM+num_MG])
    ax12.set_xticklabels([r'$\mathrm{\Gamma}$',  r'$\mathrm{K}$', r'$\mathrm{M}$', r'$\mathrm{\Gamma}$'])

    plt.legend(loc="upper right")
    plt.show()

# =============================================================================
# BC_BANDS = np.sum(FLOQUET_BC_LOOP_PATH,axis=0)
# dis = 3.364    
# lconst = 2.445   
# print(np.shape(BC_BANDS))
# 
# angle1 = np.arctan2((b*np.sqrt(3.)/2.),(a+b/2.))                               # 1st angle    
# print("angle1: "+str(angle1)) 
# angle2 = angle1 + np.pi/3.                                                     # 1st angle  
# print("angle2: "+str(angle2))
# d = np.sqrt(b**2*3./4.+(a+b/2.)**2)                                            # side length of super cell (lconst*AA)
# print("d (lconst=1): "+str(d))      
# 
# ep2x = a+b/2.+d*np.sin(np.pi/6.-angle1)                                        # x-coordinate end-point
#     
# ep2y = b*np.sqrt(3.)/2.+d*np.cos(np.pi/6.-angle1)                              # y-coordinate end-point
# 
# slope = ep2y/ep2x                                                              # slope of super cell diagonal
# 
# # superlattice bravis translational vectors
# lvec = np.zeros((2,2)) 
# lvec[0,0] =  d*np.cos(angle1)
# lvec[0,1] =  d*np.sin(angle1)
# lvec[1,0] =  d*np.sin(np.pi/6.-angle1)
# lvec[1,1] =  d*np.cos(np.pi/6.-angle1)
# 
# # Real space basis vectors of super cell
# A1 = np.array([lvec[0,0], lvec[0,1], 0.])                                      # (lconst*AA)
# A2 = np.array([lvec[1,0], lvec[1,1], 0.])                                      # (lconst*AA)
# A3 = np.array([0.0, 0.0, 1.0])*(dis/lconst)                                    # (lconst*AA)
# 
# # Reciprocal space basis vectors of super cell
# B1 = 2.*np.pi*np.cross(A2,A3)/(lconst*np.dot(A1,np.cross(A2,A3)))                       # 1/(lconst*AA)
# B2 = 2.*np.pi*np.cross(A3,A1)/(lconst*np.dot(A2,np.cross(A3,A1)))                       # 1/(lconst*AA)
# B3 = 2.*np.pi*np.cross(A1,A2)/(lconst*np.dot(A3,np.cross(A1,A2)))                       # 1/(lconst*AA)
# 
# AREA_BZ=np.linalg.norm(np.cross(B1,B2))/(64*64)
# print(AREA_BZ)
# 
# fig1 = plt.figure(2)
# gs1 = gridspec.GridSpec(1, 1)
# ax = fig1.add_subplot(gs1[0,0])
# ax.plot(BC_BANDS*AREA_BZ/(2*np.pi), 'ro')
# ax.plot(np.linspace(0,1000,1001),[0.0]*1001, 'k', linewidth=2.0)
# ax.set_xlim(981,992)
# ax.set_ylim(-6,+6)
# ax.set_xticks(np.linspace(981,992,12))
# ax.set_yticks(np.linspace(-6,6,13))
# ax.set_ylabel(r'$\mathrm{Chern}$  $\mathrm{number}$')
# ax.set_xlabel(r'$\mathrm{band}$  $\mathrm{index}$')
# plt.grid()
# plt.plot()        
#     
# =============================================================================
