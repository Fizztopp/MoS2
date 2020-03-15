/**
 *	TIGHT-BINDING MODEL FOR TWISTED BILAYER GRAPHENE (TBG)
 *  Copyright (C) 2019, Gabriel E. Topp
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2, or (at your option)
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 *  02111-1307, USA.
 * 	
 * 	This code uses a truncated (energy-cuttoff) Taylor-expanded Hamiltonian (A-->0) in initial band basis calculated by TBG_DOWNFOLDING.cc. 
 * 	Included are t.-d. circular gauge fields ([x,y]-plane) and a source-drain field in x-direction
 * 	The follwing objects can be calculated: 	
 *  -Time-dependent density
 *  -Observables: energy, longitudinal current, transversal current (Hall)	
 * 
 *  Necessary input:
 *  -Unit_Cell.dat: contains atomic positions, and sublattice index
 *  -BZ_FULL: List of k-points of Brilluoin zone
 */
 
#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <math.h>
#include <assert.h>
#include <iterator>
#include <sstream>
#include <string>
#include <algorithm>


// PARAMETERS ##########################################################

// intrinsic parameters
// electronic
#define NATOM 	1262		
#define SC        14                                                    // defines super cell (m+1,n) and thus commensurate twist angle
#define dim_new   20   

// Taylored porpagation 
#define FIRST 1.0                                                       // 1st order (1.0 on / 0.0 off)
#define SECOND 1.0                                                      // 2nd order (1.0 on / 0.0 off)
#define THIRD 1.0                                                       // 3rd order (1.0 on / 0.0 off)

// Peierls driving											 
#define w_peierls      0.0546                                            // Frequency of Applied Field (eV)
#define Ax_peierls     0.01                                              // Amplitude of Applied Field in x-direction
#define Ay_peierls     0.0                                              // Amplitude of Applied Field in y-direction
#define Az_peierls     0.0                                              // Amplitude of Applied Field in z-direction

// FLOQUET
#define m_max 2                                                         // order of truncation: m in {-m,...,0,...+m} 
#define n_max 2                                                         // order of truncation: n in {-n,...,0,...+n} (m_max == n_max!) 
#define timesteps_F 200                                                 // # of steps to perform integration over one period T=2pi/Omega     

#define PI 3.14159265359


// CALCULATION OPTIONS #################################################

#ifndef NO_MPI                                                          //REMEMBER: Each Process has its own copy of all allocated memory! --> node                                             
    #include <mpi.h>
#endif

#ifndef NO_OMP                                                          // BOTTLENECK: Diagonalization -> can't be parallelized by OpenMP
    #include <omp.h>                                                    // REMEMBER: Shared memory only on same node!
#endif
                                                                                                    	
using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name ;
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

cdouble II(0,1);

//LAPACK (Fortran 90) functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//routine to find eigensystem of Hk
extern "C" {
// computes the eigenvalues and, optionally, the left and/or right eigenvectors for HE matrices
void zheev_(char* jobz, char* uplo, int* N, cdouble* H, int* LDA, double* W, cdouble* work, int* lwork, double* rwork, int *info);
}

//INLINE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inline int fq(int i, int j, int N)
/**
 *  MAT[i,j] = Vec[fq(i,j,N)] with row index i and column index j
 */
{
    return i*N+j;
}


inline int f_FL(int m, int n, int i, int j)
/**
 *	Wrapper for Floquet matrix MAT[m, n, i, j], (2*m_max+1)x(2*n_max+1)xNATOM*NATOM block matrix element where i,j in {0,..,NATOM-1}, m in {-m,...,0,...+m}, n in {-n,...,0,...+n}
 */
{
	return (2*n_max+1)*dim_new*dim_new*m + dim_new*n + (2*n_max+1)*dim_new*i + j;
}


inline double delta(int a, int b)
/**
 *  Delta function
 */
{
	if (a==b)
		return 1.;
	else
		return 0.;
}


template <class Vec>
inline void print(Vec vec)
/**
 *	Print out vector
 */
{
	for(int i=0; i<vec.size(); i++)
		{
	    	cout << vec[i] << " ";
	    }	
	cout << endl;
}


inline double Ax_t(double time)
{
/**
 *	Peierls field for electrons in x-direction:
 *  -time: Real time coordinate
 */
    return Ax_peierls*sin(w_peierls*time);
}


inline double Ay_t(double time)
{
/**
 *	Peierls field for electrons in y-direction:
 *  -time: Real time coordinate
 */
    return Ay_peierls*cos(w_peierls*time);
}


inline double Az_t(double time)
{
/**
 *	Peierls field for electrons in z-direction:
 *  -time: real time coordinate
 */
    return Az_peierls*sin(w_peierls*time);
}


// VOID FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void ReadIn(vector<dvec> &MAT, const string& filename)
{
/**
 *	Read in real valued matrix
 */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		dvec row((istream_iterator<double>(is)),
		istream_iterator<double>());
		MAT.push_back(row);
	}
	in.close();
}


template <class Vec>
void times(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product of quadratic matrices: $C = A \cdot B$
 */
{
    int dim = sqrt(A.size());
	Vec TEMP(dim*dim);
    // Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
	    for(int j=0; j<dim; j++) {
		    TEMP[fq(j,i,dim)] = B[fq(i,j,dim)];
		   }
    }
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += A[fq(i,k,dim)]*TEMP[fq(j,k,dim)]; 
			}
		}
	}	
}


template <class Vec>
void times_dn(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of first factor: $C = A^\dagger \cdot B$
 */
{
	int dim = sqrt(A.size());
	Vec TEMP1(dim*dim);
	Vec TEMP2(dim*dim);
	// Transposition gives speed up due to avoided line break
	for(int i=0; i<dim; i++) {
		for(int j=0; j<dim; j++) {
			TEMP1[fq(j,i,dim)] = A[fq(i,j,dim)];
			TEMP2[fq(j,i,dim)] = B[fq(i,j,dim)];
		}
	}		
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
				C[fq(i,j,dim)] += conj(TEMP1[fq(i,k,dim)])*TEMP2[fq(j,k,dim)];
			}
		}
	}		
}


template <class Vec>
void times_nd(Vec &A, Vec &B, Vec &C)
/**
 *	Matrix product with Hermitian conjugation of second factor: $C = A \cdot B^\dagger$
 */
{
	int dim = sqrt(A.size());	
	for(int i=0; i<dim; ++i)
	{
		for(int j=0; j<dim; ++j)
		{
			C[fq(i,j,dim)] = 0.;
			for(int k=0; k<dim; ++k)
			{
					C[fq(i,j,dim)] += A[fq(i,k,dim)]*conj(B[fq(j,k,dim)]);
			}
		}
	}	
}


void set_Hk_DOWN(cvec &Hk_DOWN, vector<cvec*> Hk_DOWN_LIST, double time)
/**
  *	Set downfolded td Hamiltonian 
  * -dim_new: integer value of reduced leading order of Hamiltonian
 *  -Hk_DOWN: Complex vector[dim_new x dim_new] to store Hamiltonian matrix
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  * -ASD: Gauge field of source-drain field
  * -Pol: double to set chirality
  * -time: tiome variable
  */
{
	double AX = Ax_t(time);
	double AY = Ay_t(time);
	
	for(int i=0; i<dim_new*dim_new; ++i){
		Hk_DOWN[i] = (*Hk_DOWN_LIST[0])[i] + FIRST*((*Hk_DOWN_LIST[1])[i]*AX + (*Hk_DOWN_LIST[2])[i]*AY) + SECOND*1./2.*((*Hk_DOWN_LIST[3])[i]*AX*AX + 2.*(*Hk_DOWN_LIST[4])[i]*AX*AY + (*Hk_DOWN_LIST[5])[i]*AY*AY) + THIRD*1./6.*((*Hk_DOWN_LIST[6])[i]*AX*AX*AX + 3.*(*Hk_DOWN_LIST[7])[i]*AX*AX*AY + 3.*(*Hk_DOWN_LIST[8])[i]*AX*AY*AY + (*Hk_DOWN_LIST[9])[i]*AY*AY*AY); 		
	}	
}	


void set_Hk_DOWN_0(cvec &Hk_DOWN, vector<cvec*> Hk_DOWN_LIST)
/**
  *	Set downfolded td Hamiltonian 
  * -dim_new: integer value of reduced leading order of Hamiltonian
 *  -Hk_DOWN: Complex vector[dim_new x dim_new] to store Hamiltonian matrix
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  * -ASD: Gauge field of source-drain field
  * -Pol: double to set chirality
  * -time: tiome variable
  */
{
	for(int i=0; i<dim_new*dim_new; ++i){
		Hk_DOWN[i] = (*Hk_DOWN_LIST[0])[i]; 		
	}	
}	


void diagonalize_DOWN(cvec &Hk, dvec &evals_DOWN)
{
/**
 *  Diagonalization of matrix Hk. Writes eiegenvalues to vector evals and eigenvectors (normalized!) to matrix Hk
 *  -Hk: Complex vector[dim_new x dim_new] to store Hamiltonian --> transformation matrices
 * 	-evals_DOWN: Real vector[dim_new] to store eigenvalues
 *  -dim_new: integer value of reduced leading order of Hamiltonian
 */
	char    jobz = 'V';             									//'N','V':  Compute eigenvalues only/+eigenvectors
	char    uplo = 'U';              									//'U','L':  Upper/Lower triangle of H is stored
	int     matsize = dim_new;      									// The order of the matrix A.  N >= 0
	int     lda = dim_new;            									// The leading dimension of the array A.  LDA >= max(1,N)
	int     lwork = 2*dim_new-1;      									// The length of the array WORK.  LWORK >= max(1,2*N-1)
	double  rwork[3*dim_new-2];       									// dimension (max(1, 3*N-2))
	cdouble work[2*dim_new-1];        									// dimension (MAX(1,LWORK)) On exit, if INFO = 0, WORK(1) returns the optimal LWORK
	int	    info;
	zheev_(&jobz, &uplo, &matsize, &Hk[0], &lda, &evals_DOWN[0], &work[0], &lwork, &rwork[0], &info);
	assert(!info);
}	


void diagonalize_F(cvec &Hk_FLOQUET, dvec &evals_FLOQUET)
{
/**
 *  Diagonalization of Floquet matrix H_FLOQUET[(2*m_max+1) x (2*m_max+1) x NATOM x NATOM]
 *  -Hk: Complex vector[(2*m_max+1) x (2*m_max+1) x NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[(2*m_max+1) x NATOM] to store eigenvalues
 */
 	char    jobz = 'V';             									//'N','V':  Compute eigenvalues only/+eigenvectors
	char    uplo = 'U';              									//'U','L':  Upper/Lower triangle of H is stored
	int     matsize_F = dim_new*(2*n_max+1);      							// The order of the matrix A.  N >= 0
	int     lda_F = dim_new*(2*n_max+1);            							// The leading dimension of the array A.  LDA >= max(1,N)
	int     lwork_F = 2*dim_new*(2*n_max+1)-1;      							// The length of the array WORK.  LWORK >= max(1,2*N-1)
	double  rwork_F[3*dim_new*(2*n_max+1)-2];       							// dimension (max(1, 3*N-2))
	cdouble work_F[2*dim_new*(2*n_max+1)-1]; 
	int	    info;
	zheev_(&jobz, &uplo, &matsize_F, &Hk_FLOQUET[0], &lda_F, &evals_FLOQUET[0], &work_F[0], &lwork_F, &rwork_F[0], &info);
	assert(!info);
}


void ReadInMAT(vector<cvec*> Hk_DOWN_LIST, const string& filename)
{
/**
  * Read in taylore matrices from disc
  * -dim_new: integer value of reduced leading order of Hamiltonian
  * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new x dim_new] to store truncated Taylor matrices in initial band basis
  *	-filename: String to define file
  */
	ifstream in(filename);
	string record;
	if(in.fail()){
		cout << "file" << filename << "could not be found!" << endl;
	}
	while (getline(in, record))
	{
		istringstream is( record );
		cvec row((istream_iterator<cdouble>(is)),	
		istream_iterator<cdouble>());
		//cout << row.size() << " " << dim_new << "---------------------------------------------------------------------" << endl;
		for(int m=0; m<10; ++m)	
		{	
			for(int i=0; i<dim_new*dim_new; ++i)
			{
				(*Hk_DOWN_LIST[m])[i] = row[fq(m,i,dim_new*dim_new)];
				//cout << row[fq(m,i,dim_new*dim_new)];
			}
		}	

	}
	in.close();
}


void Hk_bands_DOWN(vector<dvec> &K_PATH, vector<cvec*> Hk_DOWN_LIST, const string& filename, int &numprocs, int &myrank)
/**
 *	Calculate bands of downfolded Hamiltonian path K_PATH through BZ 
 *  -K_PATH: Vector of high-symmetry path vectors
 *  -ASD: Gauge field of source-drain field
 *  -Pol: double to set chirality
 * 	-filename: String to store data
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
{
	const int num_kpoints_path = K_PATH.size();
	
	cvec Hk_DOWN(dim_new*dim_new);
	dvec evals_DOWN(dim_new);
	dvec BANDS(num_kpoints_path*dim_new);
	
	for(int k=myrank; k<num_kpoints_path; k+=numprocs)
	{
		// Read in tuncated taylor matrices from file
		ReadInMAT(Hk_DOWN_LIST, "../HK_DOWN_LIST/HK_DOWN_LIST_PATH_"+to_string(k)+".dat");
		set_Hk_DOWN_0(Hk_DOWN, Hk_DOWN_LIST);

		diagonalize_DOWN(Hk_DOWN, evals_DOWN);
					
		for(int m=0; m<dim_new; m++)
			BANDS[fq(k, m, dim_new)] = evals_DOWN[m];
	}
#ifndef NO_MPI	
	MPI_Allreduce(MPI_IN_PLACE, &BANDS[0], num_kpoints_path*dim_new, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{
		ofstream myfile (filename);
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<dim_new; m++)
				{
					myfile << BANDS[fq(k, m, dim_new)] << " " ;
				}
				myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}	
}


void Hk_bands_Floquet(dvec &BANDS_FLOQUET, dvec &OVERLAP_FLOQUET, dvec &OVERLAP_FLOQUET_s, dvec &OVERLAP_FLOQUET_p, cvec &Hk_FLOQUET, dvec &evals_FLOQUET, vector<dvec> &K_PATH, vector<cvec*> Hk_DOWN_LIST, vector<dvec> &UNIT_CELL, const dvec &lvec, int &numprocs, int &myrank)
/**
 *	Calculate Floquet bands by truncated expansion in Floquet eigenfunctions
 *  -BANDS_FLOQUET: Real vector to store Floquet eigenvalues of all k-points 
 *  -OVERLAP_FLOQUET: Real vector[num_kpoints_PATHxNATOMx(2*n_max+1)] to store overlap ov Flquet bands with equilibrium bands
 *  -Hk_FLOQUET: Complex vector[(2*m_max+1)x(2*n_max+1)xNATOMxNATOM] to store Flqoeut Hamiltonian matrix
 *  -evals_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet eigenvalues
 *  -K_PATH: vector of high-symmetry path vectors
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI) 
 */
{
	const int num_kpoints_path = K_PATH.size();
	const double T = 2.*M_PI/w_peierls;
	const double dt = T/double(timesteps_F-1);
	const double dcell = sqrt(pow(lvec[0]+lvec[2],2.)+pow(lvec[1]+lvec[3],2.));
	
	cvec *TEMP1 = new cvec(dim_new*dim_new);
	cvec *TEMP2 = new cvec(dim_new*dim_new); 
	double temp, temp_orb_s, temp_orb_p, rx, ry; 
	
	cvec H0(dim_new*dim_new);
	dvec evals(dim_new);
	cdouble tempc, temp_help, Cx, Cy, Cs; 
	
	for(int k=myrank; k<num_kpoints_path; k+=numprocs)
	{
#ifndef NO_OMP    	
	#pragma omp parallel for collapse(4)                                
#endif						                
		for(int m=-m_max; m<m_max+1; m++)
		{
			for(int n=-n_max; n<n_max+1; n++)
			{					
				for(int i=0; i<dim_new; i++)
				{
					for(int j=0; j<dim_new; j++)
					{
						Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] = 0.0;
					}
				}
			}
		}												
		if(myrank==0) cout << endl; 
		if(myrank==0) cout << "k = " << k << endl;	
		
		ReadInMAT(Hk_DOWN_LIST, "../HK_DOWN_LIST/HK_DOWN_LIST_PATH_"+to_string(k)+".dat");
		
		// Perform integration over one period T
		for(double t=0; t<T-dt/2.; t+=dt)
		{	
			if(myrank==0) cout << "time step: " << t/dt <<  endl;
			set_Hk_DOWN(TEMP1[0], Hk_DOWN_LIST, t);
			set_Hk_DOWN(TEMP2[0], Hk_DOWN_LIST, t+dt);							
			for(int m=-m_max; m<m_max+1; m++)
			{
				for(int n=-n_max; n<n_max+1; n++)
				{		
#ifndef NO_OMP    	
			#pragma omp parallel for                   
#endif								
					for(int i=0; i<dim_new; i++)
					{
						for(int j=0; j<dim_new; j++)
						{
							Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] += 0.5/T*(exp(II*w_peierls*double(m-n)*t)*(*TEMP1)[fq(i,j,dim_new)] + exp(II*w_peierls*double(m-n)*(t+dt))*(*TEMP2)[fq(i,j,dim_new)])*dt + double(m)*w_peierls*delta(i,j)*delta(m,n)/double(timesteps_F-1);
						}
					}				
				}
			}
		}
		// Diagonalize Floquet Hamiltonian in order to get eigenvalues and eigenvectors		
		diagonalize_F(Hk_FLOQUET, evals_FLOQUET);  		
		for(int jj=0; jj<dim_new*(2*n_max+1); jj++)
		{
			BANDS_FLOQUET[fq(k,jj,dim_new*(2*n_max+1))] = evals_FLOQUET[jj];
		}	
		// Calculate squared overlap of Floquet eigenstates with eigenstates of eq. Hamiltonian
		set_Hk_DOWN_0(H0, Hk_DOWN_LIST);
		diagonalize_DOWN(H0, evals);
		for(int i=0; i<dim_new*(2*n_max+1); ++i) // select Floquet band
		{
			temp = 0.;
			temp_orb_s = 0.;
			temp_orb_p = 0.;
			temp_help = 0.;
			for(int w=0; w<dim_new; ++w) // select original band
			{
				tempc = 0.;
				for(int j=0; j<dim_new; ++j) // sum over vector of length dim_new
				{
					tempc += Hk_FLOQUET[fq(i,dim_new*m_max+j,dim_new*(2*m_max+1))]*conj(H0[fq(w,j,dim_new)]);
					for(int m=-m_max; m<m_max+1; ++m) // select Floquet sector
					{
						temp_help += Hk_FLOQUET[fq(i,dim_new*(m+m_max)+j,dim_new*(2*m_max+1))]*conj(H0[fq(w,j,dim_new)]);
					}
				}	
				// Squared overlap of H(0,0) with original bands
				temp += real(tempc*conj(tempc));
				// Squared overlap with s-bands
				if(dim_new-3<w && w<dim_new)
				{
					temp_orb_s += real(temp_help*conj(temp_help));
				}
				// Squared overlap with p-bands
				if(dim_new-7<w && w<dim_new-2)	
				{	
					temp_orb_p += real(temp_help*conj(temp_help));
				}
			}
			// S-,p-weighted overlap (ala Lede) PROBLEM: truncation in energy-basis -> truncation in real space basis
			//~ for(int m=-m_max; m<m_max+1; ++m) // select Floquet sector
			//~ {
				//~ Cs = 0.;
				//~ Cx = 0.;
				//~ Cy = 0.;
				//~ for(int jj=0; jj<NATOM; ++jj) // select atomic position for calulcation pf p/s overlap
				//~ {
					//~ // Calculate position by setting center of orbital as origin
					//~ rx=UNIT_CELL[jj][0]-(lvec[0]+lvec[2])*2./3.;       
					//~ ry=UNIT_CELL[jj][1]-(lvec[1]+lvec[3])*2./3.;	

					//~ if(sqrt(rx*rx+ry*ry)<dcell/6.)
					//~ {
						//~ Cs += conj(Hk_FLOQUET[fq(i,dim_new*(m+m_max)+jj,dim_new*(2*m_max+1))])*1.0;                       // symmetric weight
						//~ Cx += conj(Hk_FLOQUET[fq(i,dim_new*(m+m_max)+jj,dim_new*(2*m_max+1))])*rx/sqrt(rx*rx+ry*ry);      // antisymemtric (px) weight
						//~ Cy += conj(Hk_FLOQUET[fq(i,dim_new*(m+m_max)+jj,dim_new*(2*m_max+1))])*ry/sqrt(rx*rx+ry*ry);      // antisymemtric (py) weight 
					//~ }
				//~ }	
				//~ temp_orb_s += real(Cs*conj(Cs));
				//~ temp_orb_p += real(Cx*conj(Cx)) + real(Cy*conj(Cy));
			//~ }		
			OVERLAP_FLOQUET[fq(k,i,dim_new*(2*n_max+1))] = temp; 
			OVERLAP_FLOQUET_s[fq(k,i,dim_new*(2*n_max+1))] = temp_orb_s; 
			OVERLAP_FLOQUET_p[fq(k,i,dim_new*(2*n_max+1))] = temp_orb_p; 
		}	  				
	}
	delete TEMP1, TEMP2;	
	
#ifndef NO_MPI		
		MPI_Allreduce(MPI_IN_PLACE, &BANDS_FLOQUET[0], dim_new*(2*n_max+1)*num_kpoints_path, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &OVERLAP_FLOQUET[0], dim_new*(2*n_max+1)*num_kpoints_path, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &OVERLAP_FLOQUET_s[0], dim_new*(2*n_max+1)*num_kpoints_path, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &OVERLAP_FLOQUET_p[0], dim_new*(2*n_max+1)*num_kpoints_path, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	// Store data
	int count = COUNT;
	if(myrank==0)
	{
		ofstream myfile ("DATA/bands_floquet_"+to_string(count)+".dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<dim_new*(2*n_max+1); m++)
				{
					myfile << BANDS_FLOQUET[fq(k,m,dim_new*(2*n_max+1))] << " " ;
				}
			myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
	if(myrank==0)
	{
		ofstream myfile ("DATA/overlap_floquet_"+to_string(count)+".dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<dim_new*(2*n_max+1); m++)
				{
					myfile << OVERLAP_FLOQUET[fq(k,m,dim_new*(2*n_max+1))] << " " ;
				}
			myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
	if(myrank==0)
	{
		ofstream myfile ("DATA/overlap_floquet_s_"+to_string(count)+".dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<dim_new*(2*n_max+1); m++)
				{
					myfile << OVERLAP_FLOQUET_s[fq(k,m,dim_new*(2*n_max+1))] << " " ;
				}
			myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
	if(myrank==0)
	{
		ofstream myfile ("DATA/overlap_floquet_p_"+to_string(count)+".dat");
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<dim_new*(2*n_max+1); m++)
				{
					myfile << OVERLAP_FLOQUET_p[fq(k,m,dim_new*(2*n_max+1))] << " " ;
				}
			myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}
}


//~ void Set_Hk_Floquet(dvec kvec, cvec &Hk_FLOQUET, vector<cvec*> Hk_DOWN_LIST)
//~ /**
 //~ *	Set Floquet Hamiltonian in k-orbital basis for use in FLOQUET_BC_LOOP()
 //~ * 	-kvec: Real vector of the reciprocal space
 //~ *  -Hk_FLOQUET: Complex vector[(2*m_max+1)x(2*n_max+1)xNATOMxNATOM] to store Flqoeut Hamiltonian matrix
 //~ *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 //~ *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 //~ */
//~ {
	//~ // PROBLME: Calculate Hk at loop points!!!
	//~ const double T = 2.*M_PI/w_peierls;
	//~ const double dt = T/double(timesteps_F-1);
	
	//~ cvec *TEMP1 = new cvec(dim_new*dim_new);
	//~ cvec *TEMP2 = new cvec(dim_new*dim_new); 
	//~ double temp; 
	
	//~ cvec H0(dim_new*dim_new);
	//~ dvec evals(dim_new);
	
	//~ int rank;
	//~ MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
//~ #ifndef NO_OMP    	
	//~ #pragma omp parallel for collapse(4)                                // PERFEKTLY nested loops are collapsed into one loop
//~ #endif						                
		//~ for(int m=-m_max; m<m_max+1; m++)
		//~ {
			//~ for(int n=-n_max; n<n_max+1; n++)
			//~ {					
				//~ for(int i=0; i<dim_new; i++)
				//~ {
					//~ for(int j=0; j<dim_new; j++)
					//~ {
						//~ Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] = 0.0;
					//~ }
				//~ }
			//~ }
		//~ }												
		//~ for(double t=0; t<T-dt/2.; t+=dt)
		//~ {	
			//~ if(rank==0) cout << "time step: " << t/dt <<  endl;
			//~ ReadInMAT(Hk_DOWN_LIST, "../HK_DOWN_LIST/HK_DOWN_LIST_PATH_"+to_string(k)+".dat");
			//~ set_Hk_DOWN(TEMP1[0], Hk_DOWN_LIST, t);
			//~ set_Hk_DOWN(TEMP2[0], Hk_DOWN_LIST, t+dt);									
			//~ for(int m=-m_max; m<m_max+1; m++)
			//~ {
				//~ for(int n=-n_max; n<n_max+1; n++)
				//~ {		
//~ #ifndef NO_OMP    	
			//~ #pragma omp parallel for                   
//~ #endif								
					//~ for(int i=0; i<dim_new; i++)
					//~ {
						//~ for(int j=0; j<dim_new; j++)
						//~ {
							//~ Hk_FLOQUET[f_FL(m+m_max, n+n_max, i, j)] += 0.5/T*(exp(II*w_peierls*double(m-n)*t)*(*TEMP1)[fq(i,j,dim_new)] + exp(II*w_peierls*double(m-n)*(t+dt))*(*TEMP2)[fq(i,j,dim_new)])*dt + double(m)*w_peierls*delta(i,j)*delta(m,n)/double(timesteps_F-1);
						//~ }
					//~ }				
				//~ }
			//~ }
		//~ }	 				
	//~ delete TEMP1, TEMP2;	
//~ }


//~ void FLOQUET_BC_LOOP(dvec kvec, double kmin, double kmax, int Nk, dvec &evals_FLOQUET, dvec &bands_BCs_FLOQUET, vector<cvec*> Hk_DOWN_LIST, const string &filename)
//~ /** 
 //~ * 	Calculate Berry curvature of expanded Floquet Hamiltonian at kvec by Berry phase over enlosed are in k-space (unit is Angstroem^2)
 //~ *  -kvec: Real vector of the reciprocal space
 //~ * 	-kmin: Double to set loop around kvec
 //~ *  -kmax: Double to set loop around kvec
 //~ * 	-Nk: Number of points pers side to perform loop 
 //~ *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 //~ *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 //~ *  -evals_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet eigenvalues
 //~ *  -bands_BCs_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet band Berry curvature
 //~ *	-filename: String to save data
 //~ */
//~ {
	//~ double dk = (kmax-kmin)/double(Nk-1);
	//~ double temp;
	//~ cdouble temp1, temp2, temp3, temp4;
	//~ dvec k0(2);	 
	//~ vector<cvec*> S_ARRAY(Nk*Nk);                                       
	//~ for(int n=0; n<Nk*Nk; n++)
		//~ S_ARRAY[n] = new cvec((2*m_max+1)*(2*n_max+1)*dim_new*dim_new);	
	
	//~ // Set k-point of lower right corner of loop
	//~ k0[0] = kvec[0]-0.5*(kmax-kmin);
	//~ k0[1] = kvec[1]-0.5*(kmax-kmin);
	
	//~ // Calculate eigenvectors of gridpoints along loop
	//~ for(int i=0; i<Nk; i++)
	//~ {
		//~ kvec[0] = k0[0]+i*dk;
		//~ for(int j=0; j<Nk; j++)
		//~ {					
			//~ kvec[1] = k0[1]+j*dk;
			//~ Set_Hk_Floquet(kvec, S_ARRAY[fq(i,j,Nk)][0], Hk_DOWN_LIST);
			//~ diagonalize_F(S_ARRAY[fq(i,j,Nk)][0], evals_FLOQUET);	
		//~ }
	//~ }
	//~ // Calculate Phase around loop
	//~ for(int n=0; n<(2*n_max+1)*dim_new; n++)	
	//~ {
		//~ bands_BCs_FLOQUET[n] = 0.;
		//~ for(int i=0; i<Nk-1; i++)
		//~ {
			//~ for(int j=0; j<Nk-1; j++)
			//~ {		
				//~ temp1 = 0.;
				//~ temp2 = 0.; 
				//~ temp3 = 0.;
				//~ temp4 = 0.; 
				//~ for(int a=0; a<(2*n_max+1)*dim_new; ++a)	
				//~ {
					//~ temp1 += conj((*S_ARRAY[fq(i,j,Nk)])[fq(n,a,(2*n_max+1)*dim_new)])*(*S_ARRAY[fq(i+1,j,Nk)])[fq(n,a,(2*n_max+1)*dim_new)];
					//~ temp2 += conj((*S_ARRAY[fq(i+1,j,Nk)])[fq(n,a,(2*n_max+1)*dim_new)])*(*S_ARRAY[fq(i+1,j+1,Nk)])[fq(n,a,(2*n_max+1)*dim_new)];
					//~ temp3 += conj((*S_ARRAY[fq(i+1,j+1,Nk)])[fq(n,a,(2*n_max+1)*dim_new)])*(*S_ARRAY[fq(i,j+1,Nk)])[fq(n,a,(2*n_max+1)*dim_new)];
					//~ temp4 += conj((*S_ARRAY[fq(i,j+1,Nk)])[fq(n,a,(2*n_max+1)*dim_new)])*(*S_ARRAY[fq(i,j,Nk)])[fq(n,a,(2*n_max+1)*dim_new)];
				//~ }
				//~ bands_BCs_FLOQUET[n] += imag(log(temp1*temp2*temp3*temp4));
			//~ }		
		//~ }	
	//~ }		
	//~ if(filename!="no_file")
	//~ {
		//~ ofstream myfile1 (filename);
		//~ if (myfile1.is_open())
		//~ {
			//~ for(int n=0; n<dim_new*(2*n_max+1); ++n) 
			//~ {
				//~ //  Berry curvature equal to phase diveded by area of loop
				//~ myfile1 << bands_BCs_FLOQUET[n]/pow(kmax-kmin,2.) << endl;
			//~ }	
			//~ myfile1.close();
		//~ }
		//~ else cout << "Unable to open file" << endl;	
	//~ }
	//~ for(int n=0; n<Nk*Nk; n++)
	//~ {                            
		//~ delete S_ARRAY[n];
	//~ }	
//~ }	


//~ void FLOQUET_BC_LOOP_PATH(double kmin, double kmax, int Nk, vector<dvec> &K_PATH, dvec &evals_FLOQUET, dvec &bands_BCs_FLOQUET, vector<cvec*> Hk_DOWN_LIST, int &numprocs, int &myrank)
//~ /** 
 //~ * 	MPI Calculation of Berry curvature of expanded Floquet Hamiltonian along high-symemtry path using FLOQUET_BC_LOOP()
 //~ * 	-kmin: Double to set loop around kvec
 //~ *  -kmax: Double to set loop around kvec
 //~ * 	-Nk: Number of points pers side to perform loop 
 //~ *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 //~ *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 //~ *  -K_PATH: Vector of high-symmetry path vectors
 //~ *  -evals_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet eigenvalues
 //~ *  -bands_BCs_FLOQUET: Real vector[(M_max+1) x NATOM] to store Floquet band Berry curvature
 //~ *	-numprocs: Total number of processes (MPI)
 //~ *	-myrank: Rank of process (MPI) 
 //~ */
//~ {
	//~ int num_kpoints = K_PATH.size();
	//~ dvec BC_ARRAY(num_kpoints*dim_new*(2*m_max+1));                                       
	
	//~ for(int k=myrank; k<num_kpoints; k+=numprocs)
	//~ {
		//~ FLOQUET_BC_LOOP(K_PATH[k], kmin, kmax, Nk, evals_FLOQUET, bands_BCs_FLOQUET, Hk_DOWN_LIST, "no_file");
		//~ for(int n=0; n<dim_new*(2*m_max+1); ++n) 
		//~ {
			//~ BC_ARRAY[fq(k,n,dim_new*(2*m_max+1))] = bands_BCs_FLOQUET[n];
		//~ }	
	//~ }	
//~ #ifndef NO_MPI		
		//~ MPI_Allreduce(MPI_IN_PLACE, &BC_ARRAY[0], dim_new*num_kpoints*(2*m_max+1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
//~ #endif	
	//~ if(myrank==0)
	//~ {	
		//~ ofstream myfile1 ("FLOQUET_BC_LOOP_PATH.dat");
		//~ if (myfile1.is_open())
		//~ {
			//~ for(int k=0; k<num_kpoints; ++k)
			//~ {
				//~ for(int n=0; n<dim_new*(2*m_max+1); ++n) 
				//~ {
					//~ myfile1 << BC_ARRAY[fq(k,n,dim_new*(2*m_max+1))]/pow(kmax-kmin,2.) << " ";
				//~ }	
				//~ myfile1 << endl;
			//~ }
			//~ myfile1.close();
		//~ }
		//~ else cout << "Unable to open file" << endl;	
	//~ }
//~ }	

// main() function #####################################################

int main(int argc, char * argv[])
{
    //************** MPI INIT ***************************
  	int numprocs=1, myrank=0, namelen;
    
#ifndef NO_MPI
  	char processor_name[MPI_MAX_PROCESSOR_NAME];
  	MPI_Init(&argc, &argv);
  	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  	MPI_Get_processor_name(processor_name, &namelen);
    
	cout << "Process " << myrank << " on " << processor_name << " out of " << numprocs << " says hello." << endl;
	MPI_Barrier(MPI_COMM_WORLD);
    
#endif
	if(myrank==0) cout << "\n\tProgram running on " << numprocs << " processors." << endl;

	//************** OPEN_MP INIT **************************************
#ifndef NO_OMP 	  
	cout << "# of processes " << omp_get_num_procs() << endl;
#pragma omp parallel 
	cout << "Thread " << omp_get_thread_num() << " out of " << omp_get_num_threads() << " says hello!" << endl;     
#endif
	//******************************************************************

	// superlattice bravis translational vectors
	const dvec lvec = { 79.4034118652, 0.0, 39.7017059326, 68.7653718225};
	
	//Read in atomic positions
	vector<dvec> UNIT_CELL;
	ReadIn(UNIT_CELL, "../Unit_Cell.dat");
	if(myrank==0) cout << "Unit_Cell.dat --> " <<  UNIT_CELL.size() << " points" << endl;
	if(NATOM != UNIT_CELL.size())
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}

	//Read in vector of k-points
	vector<dvec> K_PATH;
	ReadIn(K_PATH, "../k_path.dat");
	if(myrank==0) cout << "high-symmetry path --> " << K_PATH.size() << " points" << endl;
	int num_kpoints_PATH = K_PATH.size();
	
	// irr. BZ
	//vector of weights
	vector<dvec> kweights_irr;
	ReadIn(kweights_irr, "../k_weights_irr.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_IRR;
	ReadIn(BZ_IRR, "../k_BZ_irr.dat");
	if(myrank==0) cout << "irreducible BZ --> " << BZ_IRR.size() << " points" << endl;
	int num_kpoints_BZ = BZ_IRR.size();
	
    // full BZ
	//vector of weights
	vector<dvec> kweights_full;
	ReadIn(kweights_full, "../k_weights_full.dat");
			
	//vector of BZ vectors
	vector<dvec> BZ_FULL;
	ReadIn(BZ_FULL, "../k_BZ_full.dat");
	if(myrank==0) cout << "full BZ --> " << BZ_FULL.size() << " points" << endl;
	int num_kpoints_BZ_full = BZ_FULL.size();
	
	vector<cvec*> Hk_DOWN_LIST(10);
	for(int m=0; m<10; m++)
	{
		Hk_DOWN_LIST[m] = new cvec(dim_new*dim_new);   
	}
	
	// Berry Curvature
	dvec bands_BCs_FLOQUET(dim_new*(2*m_max+1));	
	
    // Floquet matrices 
    cvec *Hk_FLOQUET = new cvec((2*m_max+1)*(2*n_max+1)*dim_new*dim_new);  
    dvec *evals_FLOQUET = new dvec(dim_new*(2*n_max+1));  
	dvec *BANDS_FLOQUET = new dvec(num_kpoints_PATH*dim_new*(2*n_max+1));
	dvec *OVERLAP_FLOQUET = new dvec(num_kpoints_PATH*dim_new*(2*n_max+1));
	dvec *OVERLAP_FLOQUET_s = new dvec(num_kpoints_PATH*dim_new*(2*n_max+1));
	dvec *OVERLAP_FLOQUET_p = new dvec(num_kpoints_PATH*dim_new*(2*n_max+1));
	
	// CALCULATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	const clock_t begin_time = clock();                                 // time summed over all threads
#ifndef NO_OMP	 
	double dtime = omp_get_wtime();	                                    // time per core
#endif
	
	//if(myrank==0){cout << "Start caluclation of equilibrium bands" << endl;}
	Hk_bands_DOWN(K_PATH, Hk_DOWN_LIST, "DATA/bands_DOWN.dat", numprocs, myrank);

	if(myrank==0){cout << "Start caluclation of Floqeut bands along k-path" << endl;}		
	Hk_bands_Floquet(BANDS_FLOQUET[0], OVERLAP_FLOQUET[0], OVERLAP_FLOQUET_s[0], OVERLAP_FLOQUET_p[0], Hk_FLOQUET[0], evals_FLOQUET[0], K_PATH, Hk_DOWN_LIST, UNIT_CELL, lvec, numprocs, myrank);
	
	//if(myrank==0){cout << "Start caluclation of Floqeut Berry curvature along k-path" << endl;}	
	//FLOQUET_BC_LOOP_PATH( -1e-5, +1e-5, 2, K_PATH, evals_FLOQUET[0], bands_BCs_FLOQUET, Hk_DOWN_LIST, numprocs, myrank);	
	
	
	if(myrank==0)
	{ 
		cout << "Calculations time (MPI): " << float(clock() - begin_time)/CLOCKS_PER_SEC << " seconds" << endl;
#ifndef NO_OMP	
	dtime = omp_get_wtime() - dtime;
	cout << "Calculations time (OMP): " << dtime << " seconds" << endl; 
#endif	
	}
	
#ifndef NO_MPI
	MPI_Finalize();
#endif	
	
	// free memeory
	for(int m=0; m<10; m++)
	{                            
		delete Hk_DOWN_LIST[m];
	}
	delete Hk_FLOQUET;
	delete evals_FLOQUET;
	delete BANDS_FLOQUET;
	delete OVERLAP_FLOQUET;
}


