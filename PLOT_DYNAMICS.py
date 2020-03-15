# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:46:21 2018

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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
    
tc = 0.658 # 1/eV = 0.658 fs    

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['font.size'] = 14  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 24
mpl.rcParams['figure.figsize'] = [12.,12]


RED = '#e41a1c'
BLUE = '#377eb8'
GREEN = '#4daf4a'
BROWN = '#fdae61'
VIOLETT = '#6a3d9a' 

def gauss(sigma, shift, x):
    return np.exp(-0.5*((x-shift)/sigma)**2)

ii = 1                                                                         # cell index
nn = 4*(ii**2+(ii+1)*ii+(ii+1)**2)                                             # number of atoms
a = ii+1
b = ii
THETA = np.arccos((b**2+4*a*b+a**2)/(2*(b**2+a*b+a**2)))                       # THETA in Rad (angle between K and K')
print("Number of sites: "+str(nn))                                                              
print("Theta in degree: "+str(THETA*360/(2*np.pi)))

file_BANDS = open('E_t.dat','r')
E = np.loadtxt(file_BANDS)
file_BANDS.close()
file_BANDS = open('BAND_OCC_t.dat','r')
OCC = np.loadtxt(file_BANDS)
file_BANDS.close()
file_BANDS = open('C_t.dat','r')
C = np.loadtxt(file_BANDS)
file_BANDS.close()
file_BANDS = open('ASD_t.dat','r')
ASD = np.loadtxt(file_BANDS)
file_BANDS.close()

timesteps = 10000

t0 = 0
tmax = timesteps-1
time = np.linspace(0,1000,timesteps)

test=np.array([0,1,2,3,4,5])
print(test[:3])
OC_V=np.zeros(timesteps)
OC_C=np.zeros(timesteps)
for t in range(timesteps):
    for i in range(int(nn/2)):
        OC_V[t] += OCC[t,i]
        OC_C[t] += OCC[t,i+int(nn/2)]

fig1 = plt.figure(1)
gs1 = gridspec.GridSpec(4, 1)
fig1.suptitle(r'$\mathrm{(n,m) = ('+str(b)+','+str(a)+')}$, $\mathrm{\Theta = '+str(np.round(THETA*360/(2*np.pi),2))+'^\circ}$, $\mathrm{N_{atom}='+str(nn)+'}$')

ax1 = fig1.add_subplot(gs1[0,0])
ax1.plot(time, ASD[:,0]-ASD[0,0], linewidth=2.0, color=RED, label="$E_{SD}(t)$")
#ax1.plot(time, ASD[:,1]-ASD[0,1], linewidth=2.0, color=BLUE, label="$A_{SD}(t)$")
ax1.set_xlabel("$\mathrm{time}$  ($\mathrm{eV^{-1}}$)")
ax1.set_ylabel("$\mathrm{SD-Field}$ ($\mathrm{arb. units}$)")
plt.legend(loc="upper right")

ax2 = fig1.add_subplot(gs1[1,0])
ax2.plot(time, E-E[0], linewidth=2.0, color=VIOLETT, label="$E(t)$")
ax2.fill(time,abs(E[tmax]-E[0])*gauss(150., 500, time), color="gainsboro", facecolor='gainsboro',label=r"$\mathrm{pump}$")
ax2.set_xlabel("$\mathrm{time}$ ($\mathrm{eV^{-1}}$)")
ax2.set_ylabel("$\mathrm{\Delta E}$ ($\mathrm{eV}$)")
plt.legend(loc="upper right")

ax3 = fig1.add_subplot(gs1[2,0])
#ax3.plot(time, OC_V/OC_V[0], linewidth=2.0, color=BLUE, label="$V$")
ax3.plot(time, OC_C, linewidth=2.0, color=RED, label="$C$")
ax3.set_xlabel("$\mathrm{time}$ ($\mathrm{eV^{-1}}$)")
ax3.set_ylabel("$\mathrm{Occupations}$")
plt.legend(loc="upper right")

ax4 = fig1.add_subplot(gs1[3,0])
#ax.plot(E, linewidth=2.0, color=RED, label="$E(t)$")
#ax.plot(time1, np.diff(E-E[0])/np.diff(time), color=BLUE, linewidth=2.0, label="$dE/dt(t)$")
ax4.plot(time, C[:,0], linewidth=2.0, color=GREEN, label=r"$J_x(t)$")
ax4.plot(time, C[:,1], linewidth=2.0, color=VIOLETT, label=r"$J_y(t)$")
ax4.set_xlabel("$\mathrm{time}$ ($\mathrm{eV^{-1}}$)")
ax4.set_ylabel("$\mathrm{J_{i}}$ ($\mathrm{arb. units}$)")

plt.tight_layout()
plt.legend(loc="upper right")
plt.subplots_adjust(top=0.92)
plt.show()

#==============================================================================
# file_BANDS = open('BAND_OCC_t.dat','r')
# BAND_OCC = np.loadtxt(file_BANDS)
# file_BANDS.close()
#    
# 
# fig1 = plt.figure(2)
# gs1 = gridspec.GridSpec(1, 1)
# ax = fig1.add_subplot(gs1[0,0])
# 
# time = np.linspace(0,152, 5e3-1)
# fig1.suptitle(r'$\mathrm{(n,m) = ('+str(b)+','+str(a)+')}$, $\mathrm{\Theta = '+str(np.round(THETA*360/(2*np.pi),2))+'^\circ}$, $\mathrm{N_{atom}='+str(nn)+'}$, $\mathrm{A_x}=0.5$, $\mathrm{A_y}=0.5$')
# for m in range(np.size(BAND_OCC[0,:])):
#     if(m>=np.size(BAND_OCC[0,:])/2-2 and m<np.size(BAND_OCC[0,:])/2+2):
#         ax.plot(time, BAND_OCC[:,m], linewidth=1.0, label=str(m))
#     else:
#         ax.plot(time, BAND_OCC[:,m], linewidth=1.0, linestyle="--")
# ax.set_xlabel("$\mathrm{time}$ $\mathrm{in}$ $\mathrm{eV^{-1}}$")
# ax.set_ylabel("$\mathrm{Occupations}$")
# plt.legend(loc="center right")
# ax.set_ylim(-0.05,1.05)
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# plt.show()
#==============================================================================
