!  tightbinding3.f90 

	program tightbinding3

        IMPLICIT NONE
	
	complex*16,dimension(:,:),allocatable :: H1,Z1,H1b
        complex*16 H2(2,2),Z2(2,2),H2b(2,2)
	double precision,dimension(:,:),allocatable :: point,eband,kband
	double precision dd(3,3),d,lconst,lvector(3,3),K_x,ngrid,K_y
        double precision lvm(2,2),pointm(2,2),EVAL2(2)
	double precision qq,aa,t,pi,kx,ky,step,kp,dE,dgap
        double precision,dimension(:),allocatable :: EVAL1,PJ
	real Ksita_x,Ksita_y,K1_x,K1_y,K2_x,K2_y,G_x,G_y,M_x,M_y

        INTEGER ii,jj,nn,marker,j,ic,p,q,p0,q0,nband
        DOUBLE PRECISION cos_sita, sin_sita, sita, rg, kratio
        complex*16, parameter :: i=(0.d0,1.d0)
        complex*16 C
        double precision kxx,kyy,G1x,G2x,G1y,G2y,rd(2),dis
        double precision sg1x,sg1y,sg2x,sg2y,dkg,skx,sky

!-------------------------------------------------------------
! Variables related to eigenvalue solver:

      INTEGER,DIMENSION(:),ALLOCATABLE ::  IFAIL,IWORK
      INTEGER M, INFO,LWORK,iu1
!      COMPLEX*16         Z(10,10)

      DOUBLE PRECISION, ALLOCATABLE :: RWORK(:)
      COMPLEX*16, ALLOCATABLE:: WORK(:)

!-------------------------------------------------------------

*     .. External Subroutines ..
      EXTERNAL        ZHEEVX 

	pi=4.d0*datan(1.0d0)

        open(22,file='kpoints.dat')
	

        open(iu1,file='input',status='old')
        read(iu1,*) ii
        read(iu1,*) rg
        read(iu1,*) ic
        read(iu1,*) dE
        read(iu1,*) dgap
	read(iu1,*) ngrid
        read(iu1,*) nband
        read(iu1,*) kratio
        read(iu1,*) rd(1),rd(2)
        close(iu1)

	nn=2*(ii**2+(ii+1)*ii+(ii+1)**2)  ! number of atoms	

	allocate (H1(nn,nn))
	allocate (point(nn,4))
        allocate (Z1(nn,2*nband+1))
	allocate (H1b(nn,nn))
        allocate (EVAL1(2*nband+1))
        allocate (PJ(2*nband+1))
        ALLOCATE(IFAIL(nn))
        ALLOCATE(IWORK(5*nn))
        ALLOCATE(RWORK(7*nn))

        dis=3.1  !3.23
	call latticegenerator(nn,ii,point,lvector,dis)   ! call subroutine to generate atomic positions
	
!        call latticemono(pointm,lvm)

	lconst=3.161d0   ! lattice constant

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	cos_sita=(3.d0*dble(ii*ii)+3.d0*dble(ii)+0.5d0)/
     $  (3.d0*dble(ii*ii)+3.d0*dble(ii)+1.d0)
	sin_sita=sqrt(1.d0-cos_sita*cos_sita)
     
	sita=acos(cos_sita)*180.d0/pi
	print*,sita

	Ksita_x=4.d0*pi*cos_sita/(3.d0*lconst)
	Ksita_y=-4.d0*pi*sin_sita/(3.d0*lconst)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!	


	open(unit=10,file="bandbi.dat")
        open(unit=11,file="bandmo.dat")
    
!!!!!!  bandstructure: Ksita to K sampling  !!!!!!!!!!!!!!!!!!
	allocate(eband(int(ngrid)+1,2*nband+1))
	allocate(kband(int(ngrid)+1,2*nband+1))

        G1y=4.d0*pi/lvector(1,1)/dsqrt(3.d0)
        G1x=0.d0

        G2y=G1y*0.5d0
        G2x=G1y*dsqrt(3.d0)/2.d0

        K_x=(G1x+G2x)/3.d0
        K_y=(G1y+G2y)/3.d0

        M_x=G1x/2.d0
        M_y=G1y/2.d0

	G_x=0.d0
	G_y=0.d0

	K2_x=K_x
	K2_y=K_y
	K1_x=G_x
	K1_y=G_y

	step=(K2_x-K1_x)/ngrid

	marker=1

	do kx=K1_x,K2_x,step

	ky=(kx-K1_x)*(K2_y-K1_y)/(K2_x-K1_x)+K1_y
	H1=0.d0
        H2=0.d0

        call Hamiltonion(ic,dE,dgap,rg,nn,lconst,lvector,
     .       point,kx,ky,H1,H2,rd,dis)
	
!-------------bilayer------------------------------------------------------

      if(kx.eq.K1_x)then
      allocate(WORK(10))

      CALL ZHEEVX( 'N', 'I', 'U', nn, H1,nn , 0.d0,
     .     0.d0, nn-nband, nn,
     .     0.0000000001d0, M, EVAL1, Z1, nn, WORK, -1 , 
     .     RWORK, IWORK,IFAIL, INFO )

      LWORK=int(WORK(1))
      deallocate(WORK)

      allocate(WORK(LWORK))
      endif

      CALL ZHEEVX( 'N', 'I', 'U', nn, H1,nn , 0.d0,
     .     0.d0, nn-nband, nn, 
     $     0.0000000001d0, M, EVAL1, Z1, nn, WORK, LWORK , 
     $     RWORK, IWORK,IFAIL, INFO )
        
         write(*,*) 'Eigenvalues and eigenvectors created sucessfully',marker

!-------------------------------------------------------------------

	do jj=1,nband+1
	write(10,*) dsqrt((kx-K1_x)**2.d0+(ky-K1_y)**2.d0), 
     .              EVAL1(jj)
	end do

	marker=marker+1
	end do

        close(22)
        close(10)
        close(11)
	end program tightbinding3           !  End of main programm  !  


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!	lattice generator
!	generate a axb (a>b) graphene supercell lattice

	subroutine latticegenerator(nn,ii,point,lvector,dis)
	double precision point(nn,4),lvector(3,3)
	dimension ep2(1,2)
	integer c1,c2
	double precision ax,ay,e1,e2,angle1,angle2,pi,a,
     .   b,ep2x,ep2y,d,lc,dis,slope,lconst
	integer nn,ii,pp
        open(unit=7,file='poscar')

	pi=4.d0*datan(1.D0)
	lconst=3.161d0
        do i=1,3
        read(7,*) lvector(i,1),lvector(i,2),lvector(i,3)
        enddo

        do i=1,nn
        read(7,*) point(i,1),point(i,2),point(i,3)
        enddo

        close(7)


	end 


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        subroutine Hamiltonion(ic,dE,dgap,rg,nn,lconst,lvector,
     .             point,kx,ky,H1,H2,rd,dis)
	
	integer nn,ic
	complex*16 H1(nn,nn),H2(2,2)
	double precision point(nn,4)
	complex*16 EVAL(nn)
	double precision lconst,lvector(3,3),dpoint(3,3,3),dE,dgap,rg
	double precision qq1,qq2,aa1,aa2,t1,t2,kx,ky,rx,ry,lcell,rd(2)
        double precision dis
	real dd(3,3),d
	
	qq1=3.30d0/rg  ! qq1=3.15
	aa1=3.161d0
	t1=-2.7d0/rd(1)   ! qq1=-2.8 
!        t1=t1/rg
	t2=0.55d0/rd(2)  !t2=0.63
!        aa2=3.20d0  
        aa2=dis
	qq2=qq1*aa2/aa1         
	lcell=dsqrt(lvector(1,1)**2.d0+lvector(1,2)**2.d0)	
        print*,qq1
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! set up of intralayer term !!!!!!!!!!!!!!!!!!!!!!!
	do i=1,int(nn/2)                           ! bottom layer
	 do j=i+1,int(nn/2)
!!!!!!!!! find out closet distance between two atom !!!!!!!!!!!!!!!!!!!!!
	 d=lcell
	 do l=1,3
	  do k=1,3
	  dpoint(l,k,1)=(point(i,1)-point(j,1))+dble(l-2)*lvector(1,1)
     .    +dble(k-2)*lvector(2,1) !x seperation between atom i, j
	  dpoint(l,k,2)=dble(l-2)*lvector(1,2)+(point(i,2)-point(j,2))
     .    +dble(k-2)*lvector(2,2) !y seperation between atom i, j
	  dpoint(l,k,3)=(point(i,3)-point(j,3))	!z seperation
	  dd(l,k)=real(dsqrt(dpoint(l,k,1)**2.d0
     .    +dpoint(l,k,2)**2.d0+dpoint(l,k,3)**2.d0))
	  if (dd(l,k).lt.d) then
	   d=dd(l,k)
	  endif
	 enddo
	enddo

	if (d.gt.3.3d0) d=0.d0
	 do ll=1,3
	  do kk=1,3
	   dpoint(ll,kk,1)=(point(i,1)-point(j,1))+dble(ll-2)*lvector(1,1)
     .      +dble(kk-2)*lvector(2,1) !x seperation between atom i, j
	   dpoint(ll,kk,2)=dble(ll-2)*lvector(1,2)+(point(i,2)-point(j,2))
     .      +dble(kk-2)*lvector(2,2) !y seperation between atom i, j
	   dpoint(ll,kk,3)=(point(i,3)-point(j,3))	!z seperation
	   dd(ll,kk)=real(dsqrt(dpoint(ll,kk,1)**2.d0
     .      +dpoint(ll,kk,2)**2.d0+dpoint(ll,kk,3)**2.d0))
	  if (dd(ll,kk).eq.d) then
	   rx=dpoint(ll,kk,1)
	   ry=dpoint(ll,kk,2)
	   H1(i,j)= H1(i,j)+t1*dexp(qq1*(1.d0-(d/aa1)))*
     .      cdexp(dcmplx(0.d0,kx*rx+ky*ry))
	  endif
	 enddo
	enddo
	 H1(j,i)=dconjg(H1(i,j))	
	enddo
	enddo

	do i=1,int(nn/2)
		H1(i,i)=dE/2.d0   ! onsite for bottom layer
	enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	do i=int(nn/2)+1,nn         ! top layer
	 do j=i+1,nn
	 d=lcell
	  do l=1,3
	   do k=1,3
	   dpoint(l,k,1)=(point(i,1)-point(j,1))+dble(l-2)*lvector(1,1)
     .     +dble(k-2)*lvector(2,1) !x seperation between atom i, j
	   dpoint(l,k,2)=dble(l-2)*lvector(1,2)+(point(i,2)-point(j,2))
     .     +dble(k-2)*lvector(2,2) !y seperation between atom i, j
	   dpoint(l,k,3)=(point(i,3)-point(j,3))	!z seperation
	   dd(l,k)=real(dsqrt(dpoint(l,k,1)**2.d0
     .     +dpoint(l,k,2)**2.d0+dpoint(l,k,3)**2.d0))
	   if (dd(l,k).lt.d) then
	    d=dd(l,k)
	   endif
	  enddo
	 enddo

	 if (d.gt.3.3d0) d=0.d0
	  do ll=1,3
	   do kk=1,3
	   dpoint(ll,kk,1)=(point(i,1)-point(j,1))+dble(ll-2)*lvector(1,1)
     .     +dble(kk-2)*lvector(2,1) !x seperation between atom i, j
	   dpoint(ll,kk,2)=dble(ll-2)*lvector(1,2)+(point(i,2)-point(j,2))
     .     +dble(kk-2)*lvector(2,2) !y seperation between atom i, j
	   dpoint(ll,kk,3)=(point(i,3)-point(j,3))	!z seperation
	   dd(ll,kk)=real(dsqrt(dpoint(ll,kk,1)**2.d0
     .     +dpoint(ll,kk,2)**2.d0+dpoint(ll,kk,3)**2.d0))
	   if (dd(ll,kk).eq.d) then
	    rx=dpoint(ll,kk,1)
	    ry=dpoint(ll,kk,2)
	    H1(i,j)= H1(i,j)+t1*dexp(qq1*(1.d0-(d/aa1)))
     .      *cdexp( dcmplx(0.d0,kx*rx+ky*ry))
	   endif
	  enddo
	 enddo
	 H1(j,i)=dconjg(H1(i,j))
	enddo
	enddo
	
	do i=int(nn/2+1),nn
		H1(i,i)=-dE/2.d0   ! onsite for top layer
	enddo


        if (ic.eq.1) then
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! set up of interlayer term !!!!!!!!!!!!!!!!!!!!!!!!!!

	do i=1,int(nn/2)
	 do j=int(nn/2)+1,nn
!!!!!!!!! find out closet distance between two atom !!!!!!!!!!!!!!!!!!!!!
	  d=lcell
	  do l=1,3
	   do k=1,3
	   dpoint(l,k,1)=(point(i,1)-point(j,1))+dble(l-2)*lvector(1,1)
     .     +dble(k-2)*lvector(2,1) !x seperation between atom i, j
	   dpoint(l,k,2)=dble(l-2)*lvector(1,2)+(point(i,2)-point(j,2))
     .     +dble(k-2)*lvector(2,2) !y seperation between atom i, j
	   dpoint(l,k,3)=(point(i,3)-point(j,3))	!z seperation
	   dd(l,k)=real(dsqrt(dpoint(l,k,1)**2.d0
     .     +dpoint(l,k,2)**2.d0+dpoint(l,k,3)**2.d0))
	   if (dd(l,k).lt.d) then
	   d=dd(l,k)
	   endif
	  enddo
	 enddo

	 if (d.gt.7.5d0) d=0.d0
 	  do ll=1,3
	   do kk=1,3
	   dpoint(ll,kk,1)=(point(i,1)-point(j,1))+dble(ll-2)*lvector(1,1)
     .     +dble(kk-2)*lvector(2,1) !x seperation between atom i, j
	   dpoint(ll,kk,2)=dble(ll-2)*lvector(1,2)+(point(i,2)-point(j,2))
     .     +dble(kk-2)*lvector(2,2) !y seperation between atom i, j
	   dpoint(ll,kk,3)=(point(i,3)-point(j,3))	!z seperation
	   dd(ll,kk)=real(dsqrt(dpoint(ll,kk,1)**2.d0
     .     +dpoint(ll,kk,2)**2.d0+dpoint(ll,kk,3)**2.d0))
	   if (dd(ll,kk).eq.d) then
	    rx=dpoint(ll,kk,1)
	    ry=dpoint(ll,kk,2)
!	    H1(i,j)= H1(i,j)+t1*dexp(qq1*(1.d0-(d/aa1)))
!     .      *cdexp( dcmplx(0.d0,kx*rx+ky*ry))  ! Vppn term
	    H1(i,j)= H1(i,j)+t2*dexp(qq2*(1.d0-(d/aa2)))
     .      *cdexp( dcmplx(0.d0,kx*rx+ky*ry))  ! Vppq term
	   endif
	  enddo
	 enddo
	 H1(j,i)=dconjg(H1(i,j))	
	enddo
	enddo

        endif


	end
