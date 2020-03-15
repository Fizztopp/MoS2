/**
 *	TIGHT-BINDING MODEL FOR (relaxed) MoS2
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
 * 	This code is based on a full unit cell tight-binding model for twisted bilayer graphene. For commensurate angles the follwing objects can be calculated
 *  -filling-dependent chemical potential
 *  -calculation of truncated (energy-cuttoff) Taylor-expanded Hamiltonian (A-->0) in initial band basis 	
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
#define SC        14                                                    // defines super cell (m+1,n) and thus commensurate twist angle
#define NATOM     1262                   						        // # atoms (dimension of Hamiltonian)
#define lconst    3.161                                                 // lattice constant (Angstroem)                                        
#define	qq1       3.30											        // hopping renormalization 
#define	aa1       3.161												    // intralayer nearest-neigbour distance	
#define	aa2       3.10                                                   // interlayer distance (Angstroem)
#define	t1        -2.70                                                 // hopping parameter of pz-pi (eV)
#define	t2        0.55							  	                    // hopping parameter of pz-sigma (eV)
#define BETA      10.0                       					     	// inverse temperature (1/eV)
// additional options
#define rg        1.40                                                  // Fermi renormalization (1. off) <-- magic angle ~1.05 <->  Natom ~13468 <-> v_fermi ~0.0
#define VV        0.0                                                   // symmetric top-gate/back-gate potential (eV)
#define dgap      0.0                                                   // sublattice potential a la Haldane (eV)
#define rd1       -32.0                                                 // renormalization t1
#define rd2       2.45                                                  // renormalization t2

// DImension of truncated Hamiltonian
#define dim_new 20                                                       // window for energies: diemsnion of truncated Hamiltonian

#define PI 3.14159265359

// CALCULATION OPTIONS #################################################

#ifndef NO_MPI                                                          //REMEMBER: Each Process has its own copy of all allocated memory! --> node                                             
    #include <mpi.h>
#endif

#ifndef NO_OMP                                                          // BOTTLENECK: Diagonalization -> can't be parallelized by OpenMP
    #include <omp.h>                                                    // REMEMBER: Shared memory only on same node!
#endif

//#define NO_IC                                                         // switch interlayer coupling on/off       

using namespace std;

typedef complex<double> cdouble;                  						// typedef existing_type new_type_name ;
typedef vector<double> dvec;                     					    // vectors with real double values
typedef vector<cdouble> cvec;                     						// vectors with complex double values

cdouble II(0,1);


// DEFINITION OF FUNCTIONS #############################################

//LAPACK (Fortran 90) functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//routine to find eigensystem of Hk
extern "C" {
/** 
 *  Computes the eigenvalues and, optionally, the eigenvectors for a Hermitian matrices H
 */
    void zheev_(char* jobz, char* uplo, int* N, cdouble* H, int* LDA, double* W, cdouble* work, int* lwork, double* rwork, int *info);
}
//'N','V':  Compute eigenvalues only, and eigenvectors
char    jobz = 'V';       
//'U','L':  Upper, Lower triangle of H is stored 
char    uplo = 'U';  
// The order of the matrix H.  NATOM >= 0
int     matsize = NATOM;    
// The leading dimension of the array H.  lda >= max(1, NATOM)
int     lda = NATOM;             
// The length of the array work.  lwork  >= max(1,2* NATOM-1)
int     lwork = 2*NATOM-1;    
// dimension (max(1, 3* NATOM-2))
double  rwork[3*NATOM-2];  
// dimension (MAX(1,LWORK))
cdouble work[2*NATOM-1];  
// Info
int	    info;


void diagonalize(cvec &Hk, dvec &evals)
{
/**
 *  Diagonalization of matrix Hk. Stores eigenvalues in real vector evals and eigenvectors in complex vector Hk
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[NATOM] to store eigenvalues
 */
    zheev_(&jobz, &uplo, &matsize, &Hk[0], &lda, &evals[0], &work[0], &lwork, &rwork[0], &info);
	assert(!info);
}


char    jobz_eig = 'N';         
void diagonalize_eig(cvec &Hk, dvec &evals)
{
/**
 *  Diagonalization of matrix Hk. Stores ONLY eigenvalues in real vector evals
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian --> transformation matrices
 * 	-evals: Real vector[NATOM] to store eigenvalues
 */
    zheev_(&jobz_eig, &uplo, &matsize, &Hk[0], &lda, &evals[0], &work[0], &lwork, &rwork[0], &info);
	assert(!info);
}                            


//INLINE FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inline int fq(int i, int j, int N)
/**
 *  MAT[i,j] = Vec[fq(i,j,N)] with row index i and column index j
 */
{
    return i*N+j;
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


inline double fermi(double energy, double mu)
{
/**
 *	Fermi distribution:
 *	-energy: Energy eigenvalue
 *	-mu: Chemical potential
 */
    return 1./(exp((energy-mu)*BETA) + 1.);
}


inline double gauss(double time, double delay, double sigma)
/**
 *	Normalized Gauss distribution
 *	-time: time coordinate
 *	-delay: mean expectation value
 *	-sigma: standard deviation 
 **/
{
	return 1./(sigma*sqrt(2.*PI))*exp(-0.5*pow((time-delay)/sigma,2.));
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


void set_Hk0(dvec &kvec, cvec &Hk, const dvec &lvec, vector<dvec> &UNIT_CELL)
/**
 * 	Set eq. Hamiltonian (without external field)
 *  -kvec: Real vector of the reciprocal space
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
{
	const double qq2 = (qq1/rg)*aa2/aa1;
	const double kx = kvec[0];                                          // private to each Thread
	const double ky = kvec[1];                                 
		
    // Bottom layer 
#ifndef NO_OMP    
    #pragma omp parallel												// all computed values of private variables are lost after parrallel region!
	{
#endif		
	double d, rx, ry, rz;                                               // declares variable (allocates memory), private (not shared) for each thread
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 	
	for(int m=0; m<NATOM*NATOM; m++){
		Hk[m] = 0.0;
	}	
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 
	for(int i=0; i<NATOM/2; ++i)
	{
		Hk[fq(i,i,NATOM)] = VV/2.;                                      // NO mass term (V/2*sigma_3), just decreases top layer energy by const. electric potential -V/2 (V/2*sigma_0)!
		//if (UNIT_CELL[i][3] < 0.9){
        //        Hk[fq(i,i,NATOM)] += -dgap/2.;
        //}
        //else{
        //        Hk[fq(i,i,NATOM)] += dgap/2.;
        //}     
		for(int j=i+1; j<NATOM/2; ++j)
		{	
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = AA 
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = AA 
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
	                if(d<=3.30)
	                {
						Hk[fq(i,j,NATOM)] += t1/rd1*exp(qq1/rg*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry));      // [k] = 1/AA				          
					}
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);
		}
	}	
	// Top layer 
#ifndef NO_OMP    	
	#pragma omp for
#endif	  
	for(int i=NATOM/2; i<NATOM; ++i)
	{
		Hk[fq(i,i,NATOM)] = -VV/2.;                                     // NO mass term (V/2*sigma_3), just decreases top layer energy by const. electric potential -V/2 (V/2*sigma_0)!
		//if (UNIT_CELL[i][3] < 0.9){
		//	Hk[fq(i,i,NATOM)] += -dgap/2.;
        //}
        // else{
		//	Hk[fq(i,i,NATOM)] += dgap/2.;
		//}
		for(int j=i+1; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = AA 
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = AA 
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
					if(d<=3.30)
	                {
						Hk[fq(i,j,NATOM)] += t1/rd1*exp(qq1/rg*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry));      // [k] = 1/AA		
					}			  
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
	// Inter-layer terms
#ifndef NO_IC
#ifndef NO_OMP    	
	#pragma omp for
#endif
	for(int i=0; i<NATOM/2; ++i)
	{
		for(int j=NATOM/2; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = AA 
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = AA 
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
	                if(d<=7.50)
	                {
						//Hk[fq(i,j,NATOM)] += (1.-pow(aa2/d,2.))*t1/rg*exp(qq1/rg*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry));
						Hk[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry));               // Vpp_sigma term
					}	
					//  ((1.-pow(aa2/d,2.)) == 0 for vertical hopping (interlayer hopping in AA regions) --> purely governed by Vpp_sigma term  
           
				}
			}
			Hk[fq(j,i,NATOM)] = conj(Hk[fq(i,j,NATOM)]);	
		}
	}
#endif
#ifndef NO_OMP 	
	}
#endif						
}


void Hk_bands(dvec &BANDS, cvec &Hk, dvec &evals, vector<dvec> &K_PATH, vector<dvec> &UNIT_CELL, const dvec &lvec, const string& filename, int &numprocs, int &myrank)
/**
 *	Calculate bands of Hk0(k) for path K_PATH through BZ 
 *  -BANDS: Vector to store eigenvalues of all k-points 
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -evals: Vector to store eigenvalues of diagonalization
 *  -K_PATH: Vector of high-symmetry path vectors
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -lvec: Real Vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 * 	-filename: String to store data
 *	-numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
{
	const int num_kpoints_path = K_PATH.size();

	for(int k=myrank; k<num_kpoints_path; k+=numprocs)
	{
		set_Hk0(K_PATH[k], Hk, lvec, UNIT_CELL);
		diagonalize_eig(Hk, evals);
		for(int m=0; m<NATOM; m++)
			BANDS[fq(k, m, NATOM)] = evals[m];
	}
#ifndef NO_MPI	
	MPI_Allreduce(MPI_IN_PLACE, &BANDS[0], num_kpoints_path*NATOM, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif	
	if(myrank==0)
	{
		ofstream myfile (filename);
		if (myfile.is_open())
		{
			for(int k=0; k<num_kpoints_path; k++)
			{
				for(int m=0; m<NATOM; m++)
				{
					myfile << BANDS[fq(k, m, NATOM)] << " " ;
				}
				myfile  << endl;
			}
		myfile.close();
		}
		else cout << "Unable to open file" << endl;
	}	
}


void set_Hk_Taylor(dvec &kvec, vector<cvec*> Hk_Taylor, const dvec &lvec, vector<dvec> &UNIT_CELL)
/**
 *	Sets matrizes of Taylor expansion of Hk in k-orbital basis for small fields A(t)-->0
 *  -kvec: Real vector of the reciprocal space
 *  -Hk_Taylor: Vector of complex matrices[10][NATOM*NATOM] to store Taylor matrices
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 */
{
	const double qq2 = (qq1/rg)*aa2/aa1;
	const double kx = kvec[0];                                          // private to each Thread
	const double ky = kvec[1];                                      
	
    // bottom layer 
#ifndef NO_OMP    
    #pragma omp parallel												// all computed values of private variables are lost after parrallel region!
	{
#endif		
	double d, rx, ry, rz;                                               // declares variable (allocates memory), private (not shared) for each thread
#ifndef NO_OMP    	
	#pragma omp for                                 
#endif		 	
	for(int m=0; m<NATOM*NATOM; m++){
		for(int n=0; n<10; n++)	{
			(*Hk_Taylor[n])[m] = 0.0;
		}	
	 }	
#ifndef NO_OMP    	
	#pragma omp for                                  					// workload per thread is dynamic (continue in for loop)
#endif		 
	for(int i=0; i<NATOM/2; ++i)
	{
		//Back-gate voltage
		(*Hk_Taylor[0])[fq(i,i,NATOM)] = VV/2.;   
		// Sublattice potential                        
		//if (UNIT_CELL[i][3] < 0.9){
		//	(*Hk_Taylor[0])[fq(i,i,NATOM)] += -dgap/2.;
        // }
        //else{
        //    (*Hk_Taylor[0])[fq(i,i,NATOM)] += dgap/2.;
        //}   
		for(int j=i+1; j<NATOM/2; ++j)
		{	
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = AA 
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = AA 
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
	                if(d<=3.30)
					{
						// 0th order
						(*Hk_Taylor[0])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry));     
						// 1st order
						(*Hk_Taylor[1])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-II*rx);     
						(*Hk_Taylor[2])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-II*ry);  
						// 2nd order    	
						(*Hk_Taylor[3])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-rx*rx);      	
						(*Hk_Taylor[4])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-rx*ry);     	
						(*Hk_Taylor[5])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-ry*ry);   
						// 3rd order  
						(*Hk_Taylor[6])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*rx*rx*rx);    
						(*Hk_Taylor[7])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*rx*rx*ry);    
						(*Hk_Taylor[8])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*rx*ry*ry);    
						(*Hk_Taylor[9])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*ry*ry*ry);     				
					}
				}
			}
			for(int nn=0; nn<10; nn++)	{
				(*Hk_Taylor[nn])[fq(j,i,NATOM)]= conj((*Hk_Taylor[nn])[fq(i,j,NATOM)]);	
			}	
		}
	}	
	// Top layer 
#ifndef NO_OMP    	
	#pragma omp for
#endif	  
	for(int i=NATOM/2; i<NATOM; ++i)
	{
		// Top-gate voltage
		(*Hk_Taylor[0])[fq(i,i,NATOM)] = -VV/2.;         
		// Sublattice potential                              
		//if (UNIT_CELL[i][3] < 0.9){
		//	(*Hk_Taylor[0])[fq(i,i,NATOM)] += -dgap/2.;
        //}
        //else{
        //    (*Hk_Taylor[0])[fq(i,i,NATOM)] += dgap/2.;
        //}   		
		for(int j=i+1; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = AA 
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
	                if(d<=3.30)
					{
						// 0th order
						(*Hk_Taylor[0])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry));     
						// 1st order
						(*Hk_Taylor[1])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-II*rx);     
						(*Hk_Taylor[2])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-II*ry);  
						// 2nd order    	
						(*Hk_Taylor[3])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-rx*rx);      	
						(*Hk_Taylor[4])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-rx*ry);     	
						(*Hk_Taylor[5])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(-ry*ry);   
						// 3rd order  
						(*Hk_Taylor[6])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*rx*rx*rx);    
						(*Hk_Taylor[7])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*rx*rx*ry);    
						(*Hk_Taylor[8])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*rx*ry*ry);    
						(*Hk_Taylor[9])[fq(i,j,NATOM)] += t1/rd1*exp(qq1*(1.-(d/aa1)))*exp(II*(kx*rx+ky*ry))*(II*ry*ry*ry);     				
					}			           
				}
			}
			for(int nn=0; nn<10; nn++)	{
				(*Hk_Taylor[nn])[fq(j,i,NATOM)]= conj((*Hk_Taylor[nn])[fq(i,j,NATOM)]);	
			}	
		}
	}
	// Inter-layer terms 
#ifndef NO_IC
#ifndef NO_OMP    	
	#pragma omp for
#endif
	for(int i=0; i<NATOM/2; ++i)
	{
		for(int j=NATOM/2; j<NATOM; ++j)
		{
			for(int m=0; m<3; ++m)
			{
				for(int n=0; n<3; ++n)
				{
					rx = UNIT_CELL[i][0]-UNIT_CELL[j][0]+double(m-1)*lvec[0]+double(n-1)*lvec[2]; 		//[rx] = AA
					ry = double(m-1)*lvec[1]+UNIT_CELL[i][1]-UNIT_CELL[j][1]+double(n-1)*lvec[3]; 		//[ry] = AA
					rz = UNIT_CELL[i][2]-UNIT_CELL[j][2];	                                            // should be zero!
	                d = sqrt(pow(rx,2.)+pow(ry,2.)+pow(rz,2.));    //[dd] = AA 
	                if(d<=7.50)
	                {	
						// 0th order   
						(*Hk_Taylor[0])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry)); 
						// 1st order    
						(*Hk_Taylor[1])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(-II*rx);   
						(*Hk_Taylor[2])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(-II*ry); 
						// 2nd order	   
						(*Hk_Taylor[3])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(-rx*rx); 
						(*Hk_Taylor[4])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(-rx*ry); 
						(*Hk_Taylor[5])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(-ry*ry); 
						// 3nd order
						(*Hk_Taylor[6])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(II*rx*rx*rx); 
						(*Hk_Taylor[7])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(II*rx*rx*ry); 
						(*Hk_Taylor[8])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(II*rx*ry*ry); 
						(*Hk_Taylor[9])[fq(i,j,NATOM)] += t2/rd2*exp(qq2*(1.-(d/aa2)))*exp(II*(kx*rx+ky*ry))*(II*ry*ry*ry); 	           
					}
				}
			}
			for(int nn=0; nn<10; nn++)	{
				(*Hk_Taylor[nn])[fq(j,i,NATOM)]= conj((*Hk_Taylor[nn])[fq(i,j,NATOM)]);	
			}
		}
	}
#endif
#ifndef NO_OMP 	
	}
#endif					
}

	
void set_Hk_DOWN_LIST(dvec &kvec, dvec &evals, vector<cvec*> Hk_Taylor, vector<cvec*> Hk_DOWN_LIST, const dvec &lvec, vector<dvec> &UNIT_CELL, int &myrank)
/**
 * Calculates truncated Taylor matrices in intital band basis 
 * -dim_new: integer value of reduced leading order of Hamiltonian
 * -limits: integer vector[2] to store upper/lower limit regarding energy cut-off [-w_peierls*lim,w_peierls*lim)
 * -kvec: quasi momentum 
 * -evals: Real vector[NATOM] to store eigenvalues
 * -Hk_Taylor: Vector of complex matrices[10][NATOM*NATOM] to store Taylor matrices
 * -Hk_DOWN_LIST: Vector of complex matrices[10][dim_new*dim_new] to store truncated Taylor matrices in initial band basis
 * -lvec: superlattice bravis translational vectors (in lconst*Angstroem)
 * -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 * -myrank: Rank of process (MPI)
 */
 {
	if(myrank==0) cout << "Start set_Hk_DOWN_LIST() ---------------------------------------------------------------------------------------------------" << endl;
	int dimH = NATOM*NATOM;
	cvec *TEMP = new cvec(dimH); 
	cvec *SMAT = new cvec(dimH);
	
	double dtime1, dtime11;
	
	// Set band window with dim_new
	vector<int> limits(2);
	limits[0] = NATOM-dim_new;
	limits[1] = NATOM;
	
	if(myrank==0) cout << "set_Hk_DOWN_LIST(): Start diagonalization ----------------------------------------------------------------------------------" << endl; 
	set_Hk0(kvec, SMAT[0], lvec, UNIT_CELL);
	diagonalize(SMAT[0], evals);
				
	// Transform Tylor matrices to intital band basis
	for(int n=0; n<10; n++)	{
		dtime1 = omp_get_wtime();
		const clock_t begin_time1 = clock();
		times_nd(Hk_Taylor[n][0], SMAT[0], TEMP[0]);	
		dtime1 = omp_get_wtime() - dtime1;
		
		dtime11 = omp_get_wtime();
		const clock_t begin_time11 = clock();
		times(SMAT[0], TEMP[0], Hk_Taylor[n][0]);
		dtime11 = omp_get_wtime() - dtime11;
	}
	delete TEMP, SMAT;
	// Store truncated matrices
#ifndef NO_OMP 		
	#pragma omp parallel for
#endif 		
	for(int i=limits[0]; i<limits[1]; ++i)
	{
		for(int j=limits[0]; j<limits[1]; ++j)
		{
			for(int n=0; n<10; n++)	{
				(*Hk_DOWN_LIST[n])[fq(i-limits[0],j-limits[0],dim_new)] = (*Hk_Taylor[n])[fq(i,j,NATOM)]; 
			}
		}		
	}	
}	


void Calc_List(dvec &evals, cvec &Hk, vector<dvec> &BZ_FULL, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<cvec*> Hk_Taylor, vector<cvec*> Hk_DOWN_LIST, int &numprocs, int &myrank)
/**
 * Calculate bands of Hk(k) for high symmetry path and store them in "bands.txt":
 *  -evals: Real vector[NATOM] to store eigenvalues
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -BZ_FULL: k-points of reciprocal cell
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -Hk_Taylor: Vector of complex matrices[10][NATOM*NATOM] to store Taylor matrices
 *  -numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
{
	if(myrank==0)cout << "Start Calc_List()" << " --------------------------------------------------------------------------------------------------------" << endl;
	const int num_kpoints_BZ_full = BZ_FULL.size();                     
	
	dvec kvec;	
	ofstream myfile;
	
	// Propagation
	for(int k=myrank; k<num_kpoints_BZ_full; k+=numprocs)	
	{   
		// Set k-vector
		kvec = BZ_FULL[k];
		// Sets Taylor expansion of Hk in k-orbital basis
		set_Hk_Taylor(kvec, Hk_Taylor, lvec, UNIT_CELL);
        // Declare Arrays needed for propagation and storage of observables                 		
		set_Hk_DOWN_LIST(kvec, evals, Hk_Taylor, Hk_DOWN_LIST, lvec, UNIT_CELL, myrank);
		// Write matrices to file (just one row)
		myfile.open("HK_DOWN_LIST/HK_DOWN_LIST_"+to_string(k)+".dat");
		if (myfile.is_open())
		{
			for(int m=0; m<10; m++)	
			{
				for(int i=0; i<dim_new*dim_new; ++i)
				{
					myfile << (*Hk_DOWN_LIST[m])[i] << " ";
				}
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
		if(myrank==0)
		{
			cout << "k_MAT #" << k << " strored!" << endl;
		}
	}	
}


void Calc_List_PATH(dvec &evals, cvec &Hk, vector<dvec> &BZ_FULL, const dvec &lvec, vector<dvec> &UNIT_CELL, vector<cvec*> Hk_Taylor, vector<cvec*> Hk_DOWN_LIST, int &numprocs, int &myrank)
/**
 * Calculate bands of Hk(k) for high symmetry path and store them in "bands.txt":
 *  -evals: Real vector[NATOM] to store eigenvalues
 *  -Hk: Complex vector[NATOM x NATOM] to store Hamiltonian
 *  -BZ_FULL: k-points of reciprocal cell
 *  -lvec: Real vector[4] of superlattice bravis translational vectors (in lconst*Angstroem)
 *  -UNIT_CELL: Vector[NATOM] of real vectors[4] containing atomic positions and sublattice info
 *  -Hk_Taylor: Vector of complex matrices[10][NATOM*NATOM] to store Taylor matrices
 *  -numprocs: Total number of processes (MPI)
 *	-myrank: Rank of process (MPI)
 */
{
	if(myrank==0)cout << "Start Calc_List()" << " --------------------------------------------------------------------------------------------------------" << endl;
	const int num_kpoints_BZ_full = BZ_FULL.size();                     
	
	dvec kvec;		
	ofstream myfile;
	
	for(int k=myrank; k<num_kpoints_BZ_full; k+=numprocs)	
	{   
		// Set k-vector
		kvec = BZ_FULL[k];
		// Sets Taylor expansion of Hk in k-orbital basis
		set_Hk_Taylor(kvec, Hk_Taylor, lvec, UNIT_CELL);
        // Declare Arrays needed for propagation and storage of observables                 		
		set_Hk_DOWN_LIST(kvec, evals, Hk_Taylor, Hk_DOWN_LIST, lvec, UNIT_CELL, myrank);
		// Write matrices to file (just one row)
		myfile.open("HK_DOWN_LIST/HK_DOWN_LIST_PATH_"+to_string(k)+".dat");
		if (myfile.is_open())
		{
			for(int m=0; m<10; m++)	
			{
				for(int i=0; i<dim_new*dim_new; ++i)
				{
					myfile << (*Hk_DOWN_LIST[m])[i] << " ";
				}
			}	
			myfile.close();
		}
		else cout << "Unable to open file" << endl;	
		if(myrank==0)
		{
			cout << "k_MAT #" << k << " strored!" << endl;
		}
	}	
}


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
   
	// DECLARATION AND INTITALIZATIO
	const int a = SC+1;
	const int b = SC;

	if(NATOM != 2*(SC*SC+(SC+1)*SC+(SC+1)*(SC+1)))
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}
	
	// 1st angle   
	const double angle1 = atan2(double(b)*sqrt(3.)/2.,double(a)+double(b)/2.) ;
	if(myrank==0) cout << "agle1: " << angle1 << endl;
	// 2nd angle       
	const double angle2 = angle1 + PI/3. ;                                          
	if(myrank==0) cout << "agle2: " << angle2 << endl;
	
	// side length of super cell
	const double d = sqrt(double(b*b)*3./4.+pow(double(a)+double(b)/2.,2.));
	if(myrank==0) cout << "d: " << d << endl;
	
	// superlattice bravis translational vectors
	const dvec lvec = { 79.4034118652, 0.0, 39.7017059326, 68.7653718225};
	
	//Read in atomic positions
	vector<dvec> UNIT_CELL;
	ReadIn(UNIT_CELL, "Unit_Cell.dat");
	if(myrank==0) cout << "Unit_Cell.dat --> " <<  UNIT_CELL.size() << " points" << endl;
	if(NATOM != UNIT_CELL.size())
	{
		cout << "WRONG ATOMNUMBER!!! ---------------------------------------------------------------------------------------------" << endl;
		return 0;
	}
	
//	Read in vector of k-points
	vector<dvec> K_PATH;
	ReadIn(K_PATH, "k_path.dat");
	if(myrank==0) cout << "high-symmetry path --> " << K_PATH.size() << " points" << endl;
	int num_kpoints_PATH = K_PATH.size();
	
	// Irr. BZ
	// Vector of weights
	vector<dvec> kweights_irr;
	ReadIn(kweights_irr, "k_weights_irr.dat");
			
	// Vector of BZ vectors
	vector<dvec> BZ_IRR;
	ReadIn(BZ_IRR, "k_BZ_irr.dat");
	if(myrank==0) cout << "irreducible BZ --> " << BZ_IRR.size() << " points" << endl;
	int num_kpoints_BZ = BZ_IRR.size();
	
    // Full BZ
	// Vector of weights
	vector<dvec> kweights_full;
	ReadIn(kweights_full, "k_weights_full.dat");
			
	// Vector of BZ vectors
	vector<dvec> BZ_FULL;
	ReadIn(BZ_FULL, "k_BZ_full.dat");
	if(myrank==0) cout << "full BZ --> " << BZ_FULL.size() << " points" << endl;
	int num_kpoints_BZ_full = BZ_FULL.size();
	
	// Bands 
	dvec BANDS(num_kpoints_PATH*NATOM);  
	
	// Vector for eigenvalues
	dvec evals(NATOM);

	// Vector for Hamiltonian Hk
	cvec *Hk = new cvec(NATOM*NATOM);
	
	// Allocation of memory for Taylorexpansion of Hamiltonian in k-orbital basis 
	vector<cvec*> Hk_Taylor(10);                                         // !! 10 coefficient matrices for 3rd order !!
	vector<cvec*> Hk_DOWN_LIST(10);
	for(int m=0; m<10; m++)
	{
		Hk_Taylor[m] = new cvec(NATOM*NATOM);	
		Hk_DOWN_LIST[m] = new cvec(dim_new*dim_new);   
	}
	
	// CALCULATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	const clock_t begin_time = clock();                                 // time summed over all threads
#ifndef NO_OMP	 
	double dtime = omp_get_wtime();	                                    // time per core
#endif

	// Caclulation of initial bands
	Hk_bands(BANDS, Hk[0], evals, K_PATH, UNIT_CELL, lvec, "bands.dat", numprocs, myrank);
	
	// Calculate truncated matrices for taylored Hamiltonian
	Calc_List_PATH(evals, Hk[0], K_PATH, lvec, UNIT_CELL, Hk_Taylor, Hk_DOWN_LIST, numprocs, myrank);
	
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

	for(int m=0; m<10; m++)
	{                            
		delete Hk_Taylor[m];
		delete Hk_DOWN_LIST[m];
	}	
	
}


