# Python programm to calculate the super cell and its BZ for Twisted Bilayer Graphene
import numpy as np  
from scipy.linalg import expm, norm               
import spglib
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d, Axes3D

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 15
mpl.rcParams['font.size'] = 14  # <-- change fonsize globally
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.direction'] = 'inout'
mpl.rcParams['ytick.direction'] = 'inout'
mpl.rcParams['figure.titlesize'] = 28
mpl.rcParams['figure.figsize'] = [10.,10]


#### CONSTRUCT SUPER CELL ##############################################################################################################
Nmesh = 64                                                                        
num_GK = 50                                                                    # number of k-point per hgh symmetry line
num_KM = 28                                                                    # number of k-point per hgh symmetry line                                                                # interlayer distance
num_MG = 50  

file = open('parameters.dat','w')
file.write("%s " % num_GK)
file.write("\n") 
file.write("%s " % num_KM)   
file.close()

ii = 14                                                                        # cell index
nn = 2*(ii**2+(ii+1)*ii+(ii+1)**2)                                             # number of atoms
lconst = 3.161                                                                # lattice constant
dis = 3.1 
print("Number of sites: "+str(nn))

a = ii+1
b = ii


# superlattice bravis translational vectors
lvec = np.zeros((2,2)) 
lvec[0,0] =  79.4034118652
lvec[0,1] =  0.0
lvec[1,0] =  39.7017059326
lvec[1,1] =  68.7653718225


#### PLOT SUPER CELL ##############################################################################################################

# Real space basis vectors of super cell
A1 = np.array([lvec[0,0], lvec[0,1], 0.])                                      # (AA)
A2 = np.array([lvec[1,0], lvec[1,1], 0.])                                      # (AA)
A3 = np.array([0.0, 0.0, 25.2880001068])                                       # (AA)

# Reciprocal space basis vectors of super cell
B1 = 2.*np.pi*np.cross(A2,A3)/np.dot(A1,np.cross(A2,A3))                       # 1/(AA)
B2 = 2.*np.pi*np.cross(A3,A1)/np.dot(A2,np.cross(A3,A1))                       # 1/(AA)
B3 = 2.*np.pi*np.cross(A1,A2)/np.dot(A3,np.cross(A1,A2))                       # 1/(AA)

vec_diag = A1+A2
SYM_LINE = np.zeros((100,3))                                                 # reflection symmetry line

#==============================================================================

#### CALC BZ  ##############################################################################################################

THETA = np.arccos((b**2.+4.*a*b+a**2.)/(2.*(b**2.+a*b+a**2.)))                       # THETA in Rad (angle between K and K')
print("Theta in Rad: "+str(THETA))                                                                    
print("Theta in degree: "+str(THETA*360/(2.*np.pi)))

# ROTATIONS
def M(axis, theta):
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

vec_rotx = np.array([1.,0.,0.])
vec_roty = np.array([0.,1.,0.])
vec_rotz = np.array([0.,0.,1.])

cos_sita = (3.*ii*ii+3.*ii+0.5)/(3.*ii*ii+3.*ii+1.)
sin_sita = np.sqrt(1.-cos_sita*cos_sita)
     
sita=np.arccos(cos_sita)*180./np.pi
print("Theta in Rad (CHECK): "+str(sita))

#High symmetry points
Kp = np.array([1*(B1[0]+B2[0])/3., (B1[1]+B2[1])/3, 0.0]) # K' in 1/(AA) 



G1y=4.*np.pi/lvec[0,0]/np.sqrt(3.)
G1x=0.

G2y=G1y*0.5
G2x=G1y*np.sqrt(3.)/2.

#B1 = np.array([G1x,G1y,0.0])                      # 1/(AA)
#B2 = np.array([G2x,G2y,0.0])                      # 1/(AA  

K = np.array([(G1x+G2x)/3., (G1y+G2y)/3., 0.0])
print(K)

MM = np.array([G1x/2., G1y/2., 0.0])                                     # M  in 1/(AA) 
GAMMA = np.array([0.0,0.0,0.0]) # GAMMA in 1/(AA) 

rot = M(vec_rotz, 0)

# basis vectors of 1st BZ
VEC_BZ1 = 2.*(MM-GAMMA)                                                        #in 1/(lconst=2.445) 
VEC_BZ2 =  np.dot(M(vec_rotz, np.pi/3.),VEC_BZ1)                               #in 1/(lconst=2.445) 

deltaKK = np.sqrt(((K-Kp)[0])**2+((K-Kp)[1])**2.)
deltaMK = np.sqrt(((MM-GAMMA)[0])**2.+((MM-GAMMA)[1])**2.)


print(r'K1-K2 (1/AA) ='+str(np.sqrt(((K-Kp)[0])**2.+((K-Kp)[1])**2.)))

def k_path():    
    '''
    Calculates high symmetry path points and saves as k_path.dat   
    '''
    K_PATH = np.array([0.,0.,0.])
    for GK in range(num_GK):
        K_PATH = np.append(K_PATH, K_PATH[-3:]+1./num_GK*(K-GAMMA))
    for KKs in range(num_KM):
        K_PATH = np.append(K_PATH, K_PATH[-3:]+1./num_KM*(MM-K))    
    for KsG in range(num_MG-1):
        K_PATH = np.append(K_PATH, K_PATH[-3:]+1/num_GK*(GAMMA-(K+(MM-K))))    
 
    K_PATH = K_PATH.reshape(int(np.size(K_PATH)/3),3)                # Array of k-vectors of shape (6*K_num+1, 3) 
    num_kpoints = np.size(K_PATH[:,0])
    print("Number of kpoints: " + str(num_kpoints) + " (path)")
    file = open('k_path.dat','w')
    for i in range(num_kpoints):
        K_PATH[i][:] = np.dot(rot,K_PATH[i][:])
        for j in range(3):
            file.write("%s " % K_PATH[i][j].real)
        file.write("\n")    
    file.close()
    return K_PATH

K_PATH = k_path()


def k_irr_BZ():    
    '''
    Calculates the k-vectors of the irreducable/full BZ
    '''
    
    mesh = [Nmesh, Nmesh, 1]    
    lattice = np.array([A1, A2, A3])                                           #(lconst=1)             
    positions = [[0.0, 0.0, 0.0]]    
    numbers= [1]             
             
    cell = (lattice, positions, numbers)
    
    print('spacegroup: ' +str(spglib.get_spacegroup(cell, symprec=1e-5)))
    #print(spglib.get_symmetry(cell, symprec=1e-5))    
    
    # caclulatio of irr. BZ vectors + weights
    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0.,0.,0.])
    MAT_help = grid[np.unique(mapping)]/np.array(mesh, dtype=float)
    MAT_irr_BZ = np.zeros((np.size(MAT_help[:,0]),3))       
    for k in range(1,np.size(MAT_help[:,0])):
        MAT_irr_BZ[k,:] = B1*MAT_help[k,0] + B2*MAT_help[k,1] + B3*MAT_help[k,2] # transform from fractional to cartesian coordinates

    print("Number of kpoints: %d (irr BZ)" % len(np.unique(mapping)))
    num_kpoints = np.size(MAT_irr_BZ[:,0])

    weights = (np.unique(mapping,return_counts=True)[1])
    print("Number of kpoints: %d (full BZ, check of weights)" % weights.sum())           
           
    #for i, (ir_gp_id, gp) in enumerate(zip(mapping, grid)):
    #    print("%3d ->%3d %s" % (i, ir_gp_id, gp.astype(float) / mesh))
    #print(grid[np.unique(mapping)]/np.array(mesh, dtype=float))    
    #print(np.unique(mapping,return_counts=True))      
    
    # caclulatio of full BZ vectors (weights = 1) 
    MAT_BZ_full = np.array(grid/np.array(mesh, dtype=float), dtype=float)
    for k in range(1,np.size(MAT_BZ_full[:,0])):
        MAT_BZ_full[k,:] = B1*MAT_BZ_full[k,0] + B2*MAT_BZ_full[k,1] + B3*MAT_BZ_full[k,2]
    print("Number of kpoints: %d (full BZ)" % np.size(MAT_BZ_full[:,0]))
    
    file = open('k_BZ_irr.dat','w')
    for i in range(num_kpoints):
        for j in range(3):
            file.write("%s " % (MAT_irr_BZ[i][j]))
        file.write("\n")    
    file.close()
    
    file = open('k_weights_irr.dat','w')
    for i in range(num_kpoints):
        file.write("%s " % (weights[i]*1.0))
        file.write("\n")    
    file.close()
    
    file = open('k_BZ_full.dat','w')
    for i in range(np.size(MAT_BZ_full[:,0])):
        for j in range(3):
            file.write("%s " % (MAT_BZ_full[i][j]))
        file.write("\n")    
    file.close()
    
    file = open('k_weights_full.dat','w')
    for i in range(np.size(MAT_BZ_full[:,0])):
        file.write("%s " % 1.0)
        file.write("\n")    
    file.close()
    return MAT_irr_BZ, MAT_BZ_full               # in 1/(AA)   


MAT_irr_BZ, MAT_BZ_full = k_irr_BZ()
     
fig = plt.figure(3)
fig.suptitle(r'$\mathrm{(n,m) = ('+str(b)+','+str(a)+')}$, $\mathrm{\Theta = '+str(np.round(THETA*360/(2*np.pi),2))+'^\circ}$, $\mathrm{N_{atom}='+str(nn)+'}$')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(K_PATH[:,0], K_PATH[:,1], K_PATH[:,2], c="b", marker="x", label="symmetry path")
ax.scatter(MAT_irr_BZ[:,0], MAT_irr_BZ[:,1], MAT_irr_BZ[:,2], c="r", marker="x", label="irr. BZ")
ax.scatter(MAT_BZ_full[:,0], MAT_BZ_full[:,1], MAT_BZ_full[:,2], c="k", marker=".", label="full BZ")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")
ax.set_xlim(-0.08,0.08)
ax.set_ylim(-0.08,0.08)
#ax.scatter(B1[0], B1[1], c="g", marker="o")
#ax.scatter(B2[0], B2[1], c="g", marker="o")
ax.scatter(G1x, G1y, c="y", marker="o")
ax.scatter(G2x, G2y, c="y", marker="o")
ax.scatter(Kp[0], Kp[1], c="y", marker="o")
plt.legend()
plt.show()                                   

