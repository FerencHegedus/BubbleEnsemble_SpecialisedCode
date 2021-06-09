#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <ctime>
#include <random>
#include <math.h>
#include "cuda_profiler_api.h"

#define PI 3.14159265358979323846

#define CLUSTER_SIZE 32
#define BUBBLE_CONSTANTS 21
#define FREQUENCY_RESOLUTION 1
#define AMPLITUDE_RESOLUTION 480
#define INITIAL_CONDITIONS 1

#define FMIN 1000.0
#define FMAX 2000.0
#define PAMIN 0.0
#define PAMAX 5.0
#define RMIN 0.8
#define RMAX 1.5
#define PERIOD 1.0

#define RELTOL 1e-9
#define ABSTOL 1e-9
#define LINTOL 1e-9

#define LINSOLV 0

using namespace std;

void ListCUDADevices();
int  SelectDeviceByClosestRevision(int, int);
void PrintPropertiesOfSpecificDevice(int);

void Linspace(double*, double, double, int);
void Logspace(double*, double, double, int);

void LoadBubbleSizes(double*, string, int);
void LoadCouplingMatrix(double*, string, int);
void SetInitialConditionsAndParameters(double*, double*, double*, double*, double*, double*);

__global__ void Integrator(double*, double*, double*, const double* __restrict__, int);
__forceinline__ __device__ double OdeFunction(double*, double, double*, double*, double&, double*, double*, double&, const double* __restrict__);
__forceinline__ __device__ void LinearEquationSolver(double*, double, double&, double*, double*, double, const double* __restrict__);
__forceinline__ __device__ void BlockReductionOmega(double*, double&);
__forceinline__ __device__ void BlockReductionMin(double*);
__forceinline__ __device__ void BlockReductionDotProduct(double*, double*, double*, double&);
__forceinline__ __device__ void BlockReductionSum(double*);

int main()
{
	// SELECT CUDA DEVICE -----------------------
	ListCUDADevices();
	
	int MajorRevision  = 3;
	int MinorRevision  = 5;
	int SelectedDevice = SelectDeviceByClosestRevision(MajorRevision, MinorRevision);
	
	PrintPropertiesOfSpecificDevice(SelectedDevice);
	
	// DATA CONFIGURATION HOST ------------------
	const int ParameterCombinations = FREQUENCY_RESOLUTION*AMPLITUDE_RESOLUTION*INITIAL_CONDITIONS;
	double* h_StateX1    = (double*)aligned_alloc(64, CLUSTER_SIZE * ParameterCombinations * sizeof(double));
	double* h_StateX2    = (double*)aligned_alloc(64, CLUSTER_SIZE * ParameterCombinations * sizeof(double));
	double* h_Parameters = (double*)aligned_alloc(64, CLUSTER_SIZE * ParameterCombinations * BUBBLE_CONSTANTS * sizeof(double));
	
	double* Frequencies      = (double*)aligned_alloc(64, FREQUENCY_RESOLUTION * sizeof(double));
	double* Amplitudes       = (double*)aligned_alloc(64, AMPLITUDE_RESOLUTION * sizeof(double));
	double* BubbleSizes      = (double*)aligned_alloc(64, CLUSTER_SIZE * sizeof(double));
	double* h_CouplingMatrix = (double*)aligned_alloc(64, CLUSTER_SIZE * CLUSTER_SIZE * sizeof(double)); //Column major
	
	Logspace(Frequencies, FMIN, FMAX, FREQUENCY_RESOLUTION);
	Linspace(Amplitudes, PAMIN, PAMAX, AMPLITUDE_RESOLUTION);
	
	LoadBubbleSizes(BubbleSizes, "BubbleSizes_N32_D1_d150_RE0.8-5.txt", CLUSTER_SIZE);
	LoadCouplingMatrix(h_CouplingMatrix, "CouplingMatrix_N32_D1_d150_RE0.8-5.txt", CLUSTER_SIZE);
	SetInitialConditionsAndParameters(h_StateX1, h_StateX2, h_Parameters, Frequencies, Amplitudes, BubbleSizes);
	
	// DATA CONFIGURATION DEVICE ----------------
	double* d_StateX1;
	double* d_StateX2;
	double* d_Parameters;
	double* d_CouplingMatrix;
	cudaMalloc((void**)&d_StateX1, CLUSTER_SIZE * ParameterCombinations * sizeof(double));
	cudaMalloc((void**)&d_StateX2, CLUSTER_SIZE * ParameterCombinations * sizeof(double));
	cudaMalloc((void**)&d_Parameters, CLUSTER_SIZE * ParameterCombinations * BUBBLE_CONSTANTS * sizeof(double));
	cudaMalloc((void**)&d_CouplingMatrix, CLUSTER_SIZE * CLUSTER_SIZE * sizeof(double));
	
	cudaMemcpy(d_StateX1, h_StateX1, CLUSTER_SIZE * ParameterCombinations * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_StateX2, h_StateX2, CLUSTER_SIZE * ParameterCombinations * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Parameters, h_Parameters, CLUSTER_SIZE * ParameterCombinations * BUBBLE_CONSTANTS * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_CouplingMatrix, h_CouplingMatrix, CLUSTER_SIZE * CLUSTER_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	
	// INTEGRATION ------------------------------
	int GridSize = ParameterCombinations;
	int BlockSize = CLUSTER_SIZE;
	int NumberOfThreads = CLUSTER_SIZE * ParameterCombinations;
	
	clock_t SimulationStart = clock();
	Integrator<<<GridSize,BlockSize>>>(d_StateX1, d_StateX2, d_Parameters, d_CouplingMatrix, NumberOfThreads);
	cudaDeviceSynchronize();
	clock_t SimulationEnd = clock();
	cout << "Simulation time: " << 1000.0*(SimulationEnd-SimulationStart) / CLOCKS_PER_SEC << "ms" << endl << endl;
	
	cudaMemcpy(h_StateX1, d_StateX1, CLUSTER_SIZE * ParameterCombinations * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_StateX2, d_StateX2, CLUSTER_SIZE * ParameterCombinations * sizeof(double), cudaMemcpyDeviceToHost);
	
	
	// SAVA DATA --------------------------------
	ofstream DataFile;
	DataFile.open ( "TestData.txt" );
	
	int Width = 20;
	DataFile.precision(13);
	DataFile.flags(ios::scientific);
	
	for (int tid=0; tid<CLUSTER_SIZE*ParameterCombinations; tid++)
	{
		DataFile.width(Width); DataFile << h_Parameters[tid+15*NumberOfThreads] << ',';
		DataFile.width(Width); DataFile << h_Parameters[tid+16*NumberOfThreads] << ',';
		DataFile.width(Width); DataFile << h_StateX1[tid] << ',';
		DataFile.width(Width); DataFile << h_StateX2[tid] << ',';
		DataFile << '\n';
	}
	
	DataFile.close();
}

// GPU SOLVER -------------------------------------------------------------------------------------

__global__ void Integrator(double* g_StateX1, double* g_StateX2, double* g_Parameters, const double* __restrict__ g_CouplingMatrix, int NT)
{
	// THREAD MANAGEMENT
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	// REGISTER MANAGEMENT
	double rX[2];
	double rP[13];
	
	double rk1[2];
	double rk2[2];
	double rk3[2];
	double rk4[2];
	double rk5[2];
	double rk6[2];
	double rx[2];
	
	double rb;
	double rD1;
	
	double rError;
	double rTolerance;
	double rRelError;
	
	rX[0] = g_StateX1[tid];
	rX[1] = g_StateX2[tid];
	
	int ParameterShift = CLUSTER_SIZE*FREQUENCY_RESOLUTION*AMPLITUDE_RESOLUTION*INITIAL_CONDITIONS;
	for (int i=0; i<13; i++)
		rP[i] = g_Parameters[tid+i*ParameterShift];
	
	// SHARED MEMORY MANAGEMENT
	//__shared__ double sD1[CLUSTER_SIZE];
	__shared__ double sD2[CLUSTER_SIZE];
	__shared__ double sdx2[CLUSTER_SIZE];
	
	__shared__ double sT;
	__shared__ double st;
	__shared__ double sdt;
	__shared__ double sO;
	__shared__ double sTimeStepMultiplicator;
	__shared__ int sStopCondition;
	
	if ( threadIdx.x == 0 )
	{
		sT = 0.0;
		sdt = 1e-3;
		sStopCondition = 0;
	}
	
	while ( sStopCondition == 0 )
	{
		// CHECK END
		if ( threadIdx.x == 0 )
		{
			if ( sT+sdt > PERIOD )
			{
				sdt = PERIOD-sT;
				sStopCondition = 1;
			}
		}
		
		// STAGES
		for (int i=0; i<2; i++)
			rx[i] = rX[i];
		if ( threadIdx.x == 0 )
			st = sT;
		__syncthreads();
		rb = OdeFunction(rk1, st, rx, rP, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		LinearEquationSolver(rk1, rb, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		
		for (int i=0; i<2; i++)
			rx[i] = rX[i] + (1.0/5.0)*rk1[i]*sdt;
		if ( threadIdx.x == 0 )
			st = sT + (1.0/5.0)*sdt;
		__syncthreads();
		rb = OdeFunction(rk2, st, rx, rP, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		LinearEquationSolver(rk2, rb, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		
		for (int i=0; i<2; i++)
			rx[i] = rX[i] + ( (3.0/40.0)*rk1[i] + (9.0/40.0)*rk2[i] )*sdt;
		if ( threadIdx.x == 0 )
			st = sT + (3.0/10.0)*sdt;
		__syncthreads();
		rb = OdeFunction(rk3, st, rx, rP, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		LinearEquationSolver(rk3, rb, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		
		for (int i=0; i<2; i++)
			rx[i] = rX[i] + ( (3.0/10.0)*rk1[i] + (-9.0/10.0)*rk2[i] + (6.0/5.0)*rk3[i] )*sdt;
		if ( threadIdx.x == 0 )
			st = sT + (3.0/5.0)*sdt;
		__syncthreads();
		rb = OdeFunction(rk4, st, rx, rP, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		LinearEquationSolver(rk4, rb, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		
		for (int i=0; i<2; i++)
			rx[i] = rX[i] + ( (-11.0/54.0)*rk1[i] + (5.0/2.0)*rk2[i] + (-70.0/27.0)*rk3[i] + (35.0/27.0)*rk4[i] )*sdt;
		if ( threadIdx.x == 0 )
			st = sT + sdt;
		__syncthreads();
		rb = OdeFunction(rk5, st, rx, rP, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		LinearEquationSolver(rk5, rb, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		
		for (int i=0; i<2; i++)
			rx[i] = rX[i] + ( (1631.0/55296.0)*rk1[i] + (175.0/512.0)*rk2[i] + (575.0/13824.0)*rk3[i] + (44275.0/110592.0)*rk4[i] + (253.0/4096.0)*rk5[i] )*sdt;
		if ( threadIdx.x == 0 )
			st = sT + (7.0/8.0)*sdt;
		__syncthreads();
		rb = OdeFunction(rk6, st, rx, rP, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		LinearEquationSolver(rk6, rb, rD1, sD2, sdx2, sO, g_CouplingMatrix);
		
		// ERROR CONTROLL
		sD2[threadIdx.x] = 1e30; // Temporary storage for relative error
		for (int i=0; i<2; i++)
		{
			rx[i] = rX[i] + ( (37.0/378.0)*rk1[i] + (250.0/621.0)*rk3[i] + (125.0/594.0)*rk4[i] + (512.0/1771.0)*rk6[i] )*sdt;
			rError = (37.0/378.0-2825.0/27648.0)*rk1[i] + (250.0/621.0-18575.0/48384.0)*rk3[i] + (125.0/594.0-13525.0/55296.0)*rk4[i] + (-277.0/14336.0)*rk5[i] + (512.0/1771.0-1.0/4.0)*rk6[i];
			rError = sdt*fabs(rError) + 1e-18;
			rTolerance = fmax( RELTOL*fmax( fabs(rx[i]), fabs(rX[i]) ), ABSTOL );
			
			sD2[threadIdx.x] = fmin( sD2[threadIdx.x], rTolerance / rError );
		}
		__syncthreads();
		
		if ( threadIdx.x < 32 )
			BlockReductionMin(sD2);
		
		if ( threadIdx.x == 0 )
		{
			if ( sD2[0] > 1 )
			{
				sTimeStepMultiplicator = 0.8*pow(sD2[0], (1.0/5.0));
				if ( isfinite(sTimeStepMultiplicator) == 1 )
					sT = sT + sdt;
			}
			else
				sTimeStepMultiplicator = 0.8*pow(sD2[0], (1.0/4.0));
			
			if ( isfinite(sTimeStepMultiplicator) == 0 )
			{
				sTimeStepMultiplicator = 0.1;
				sD2[0] == 0.5;
			}
			
			sTimeStepMultiplicator = fmin(sTimeStepMultiplicator, 5.0);
			sTimeStepMultiplicator = fmax(sTimeStepMultiplicator, 0.1);
			
			sdt = sdt * sTimeStepMultiplicator;
			
			sdt = fmin(sdt, 1.0);
			sdt = fmax(sdt, 1e-12);
		}
		__syncthreads();
		
		// UPDATE STATE
		if ( sD2[0] > 1 )
		{
			for (int i=0; i<2; i++)
				rX[i] = rx[i];
		}
	}
	
	// STORE RESULTS BACK
	g_StateX1[tid] = rX[0];
	g_StateX2[tid] = rX[1];
}

__forceinline__ __device__ double OdeFunction(double* F, double t, double* x, double* p, double& rD1, double* sD2, double* sdx2, double& sO, const double* __restrict__ A)
{
	sD2[threadIdx.x] = x[0]; // Temporary storage of bubble radii in sD2
	__syncthreads();
	
	if ( threadIdx.x < 32 )
		BlockReductionOmega(sD2, sO);
	
	double rx1 = 1.0/x[0];
	double pg  = pow(rx1, p[10]);
	
	double s1;
	double c1;
	sincospi(2.0*t, &s1, &c1);
	
	double s2 = sin(2.0*p[11]*PI*t+p[12]);
	double c2 = cos(2.0*p[11]*PI*t+p[12]);
	
	double N;
	double D;
	double rD;
	
	N = (p[0]+p[1]*x[1])*pg - p[2]*(1.0+p[9]*x[1]) - p[3]*rx1 - p[4]*x[1]*rx1 - (1.5-0.5*p[9]*x[1])*x[1]*x[1] - ( p[5]*s1 + p[6]*s2 ) * (1.0+p[9]*x[1]) - x[0]*(p[7]*c1 + p[8]*c2);
	D = x[0] - p[9]*x[0]*x[1] + p[4]*p[9];
	rD = 1.0/D;
	
	// Uncoupled part
	F[0] = x[1];
	sdx2[threadIdx.x] = N*rD; // Initial guess for dx2
	
	// Coupled part
	__shared__ double cpt[CLUSTER_SIZE];
	cpt[threadIdx.x] = -2.0*x[0]*x[1]*x[1];
	__syncthreads();
	
	// For linear solver
	rD1 = rD;
	sD2[threadIdx.x] = x[0]*x[0];
	
	double b = 0;
	for (int i=0; i<CLUSTER_SIZE; i++)
		b = b + A[threadIdx.x+i*CLUSTER_SIZE]*cpt[i];
	
	b = (b+N)*rD;
	return b;
}

__forceinline__ __device__ void LinearEquationSolver(double* F, double b, double& rD1, double* sD2, double* sdx2, double sO, const double* __restrict__ A)
{
	double sum;
	double error;
	int tstop = 0;
	__shared__ int bstop;
	
	if ( threadIdx.x == 0 )
		bstop = 0;
	__syncthreads();
	
	// MODIFIED JACOBI ITERATION
	#if LINSOLV == 0
		int Counter = 0;
		while ( bstop < CLUSTER_SIZE )
		{
			sum = 0;
			for (int i=0; i<CLUSTER_SIZE; i++)
				sum = sum + A[threadIdx.x+i*CLUSTER_SIZE]*sD2[i]*sdx2[i];
			
			sum = sum*rD1;
			
			F[1] = sO*(b-sum) + (1.0-sO)*sdx2[threadIdx.x];
			__syncthreads();
			
			error = abs( sdx2[threadIdx.x] - F[1] );
			sdx2[threadIdx.x] = F[1];
			
			if ( ( error < LINTOL ) && ( tstop == 0 ) )
			{
				atomicAdd(&bstop,1);
				tstop = 1;
			}
			__syncthreads();
			
			Counter++;
		}
	#endif
	
	// BICONJUGATE GRADIENT METHOD
	#if LINSOLV == 1
		double val;
		double rqt;
		
		__shared__ double sr[CLUSTER_SIZE];
		__shared__ double srt[CLUSTER_SIZE];
		__shared__ double sp[CLUSTER_SIZE];
		__shared__ double spt[CLUSTER_SIZE];
		__shared__ double sq[CLUSTER_SIZE];
		__shared__ double tmp[CLUSTER_SIZE];
		__shared__ double srho1;
		__shared__ double srho2;
		__shared__ double salpha;
		__shared__ double sbeta;
		
		// Initial residual
		sum = 0;
		for (int i=0; i<CLUSTER_SIZE; i++)
			sum = sum + A[threadIdx.x+i*CLUSTER_SIZE]*sD2[i]*sdx2[i];
		
		sum = sum*rD1 + sdx2[threadIdx.x];
		
		sr[threadIdx.x] = b-sum;
		srt[threadIdx.x] = sr[threadIdx.x];
		__syncthreads();
		
		// Iteration
		int Counter = 0;
		while ( bstop < CLUSTER_SIZE )
		{
			BlockReductionDotProduct(sr, srt, tmp, srho1);
			
			if ( Counter == 0 )
			{
				sp[threadIdx.x] = sr[threadIdx.x];
				spt[threadIdx.x] = srt[threadIdx.x];
				
				if ( fabs(srho1) < 1e-6 )
					break;
			} else
			{
				if ( threadIdx.x == 0 )
					sbeta = srho1/srho2;
				__syncthreads();
				
				sp[threadIdx.x] = sr[threadIdx.x] + sbeta*sp[threadIdx.x];
				spt[threadIdx.x] = srt[threadIdx.x] + sbeta*spt[threadIdx.x];
			}
			__syncthreads();
			
			sum = 0;
			for (int i=0; i<CLUSTER_SIZE; i++)
			{
				val = rD1*A[threadIdx.x+i*CLUSTER_SIZE]*sD2[i];
				sum = sum + val*sp[i];
				tmp[threadIdx.x] = spt[threadIdx.x]*val;
				__syncthreads();
				
				BlockReductionSum(tmp);
				
				if ( threadIdx.x == i )
					rqt = spt[threadIdx.x] + tmp[0];
				__syncthreads();
			}
			sq[threadIdx.x] = sum + sp[threadIdx.x];
			__syncthreads();
			
			BlockReductionDotProduct(spt, sq, tmp, salpha); // salpha is dummy variable
			
			if ( threadIdx.x == 0 )
				salpha = srho1 / tmp[0];
			__syncthreads();
			
			F[1] = sdx2[threadIdx.x] + salpha*sp[threadIdx.x];
			sr[threadIdx.x] = sr[threadIdx.x] - salpha*sq[threadIdx.x];
			srt[threadIdx.x] = srt[threadIdx.x] - salpha*rqt;
			__syncthreads();
			
			if ( threadIdx.x == 0 )
				srho2 = srho1;
			
			error = abs( sdx2[threadIdx.x] - F[1] );
			sdx2[threadIdx.x] = F[1];
			
			if ( ( error < LINTOL ) && ( tstop == 0 ) )
			{
				atomicAdd(&bstop,1);
				tstop = 1;
			}
			__syncthreads();
			
			Counter++;
		}
	#endif
}

__forceinline__ __device__ void BlockReductionOmega(double* x, double& sO)
{
	// Block reduction
	for (int i=0; i<(CLUSTER_SIZE/32 - 1); i++)
		x[threadIdx.x] = ( x[threadIdx.x] > x[threadIdx.x+i*CLUSTER_SIZE] ) ? x[threadIdx.x] : x[threadIdx.x+i*CLUSTER_SIZE]; __syncwarp();
	
	// Warp reduction
	x[threadIdx.x] = ( x[threadIdx.x] > x[threadIdx.x+16] ) ? x[threadIdx.x] : x[threadIdx.x+16]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] > x[threadIdx.x+ 8] ) ? x[threadIdx.x] : x[threadIdx.x+ 8]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] > x[threadIdx.x+ 4] ) ? x[threadIdx.x] : x[threadIdx.x+ 4]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] > x[threadIdx.x+ 2] ) ? x[threadIdx.x] : x[threadIdx.x+ 2]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] > x[threadIdx.x+ 1] ) ? x[threadIdx.x] : x[threadIdx.x+ 1]; __syncwarp();
	
	if ( threadIdx.x == 0 )
		sO = __expf(-0.04*x[0])*0.5 + 0.5;
}

__forceinline__ __device__ void BlockReductionMin(double* x)
{
	// Block reduction
	for (int i=0; i<(CLUSTER_SIZE/32 - 1); i++)
		x[threadIdx.x] = ( x[threadIdx.x] < x[threadIdx.x+i*CLUSTER_SIZE] ) ? x[threadIdx.x] : x[threadIdx.x+i*CLUSTER_SIZE]; __syncwarp();
	
	// Warp reduction
	x[threadIdx.x] = ( x[threadIdx.x] < x[threadIdx.x+16] ) ? x[threadIdx.x] : x[threadIdx.x+16]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] < x[threadIdx.x+ 8] ) ? x[threadIdx.x] : x[threadIdx.x+ 8]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] < x[threadIdx.x+ 4] ) ? x[threadIdx.x] : x[threadIdx.x+ 4]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] < x[threadIdx.x+ 2] ) ? x[threadIdx.x] : x[threadIdx.x+ 2]; __syncwarp();
	x[threadIdx.x] = ( x[threadIdx.x] < x[threadIdx.x+ 1] ) ? x[threadIdx.x] : x[threadIdx.x+ 1]; __syncwarp();
}

__forceinline__ __device__ void BlockReductionDotProduct(double* x1, double* x2, double* tmp, double& y)
{
	tmp[threadIdx.x] = x1[threadIdx.x] * x2[threadIdx.x];
	__syncthreads();
	
	for (int i=blockDim.x/2; i>0; i>>=1)
	{
		if ( threadIdx.x < i )
			tmp[threadIdx.x] += tmp[threadIdx.x + i];
		__syncthreads();
	}
	__syncthreads();
	
	if ( threadIdx.x == 0 )
		y = tmp[0];
}

__forceinline__ __device__ void BlockReductionSum(double* tmp)
{
	for (int i=blockDim.x/2; i>0; i>>=1)
	{
		if ( threadIdx.x < i )
			tmp[threadIdx.x] += tmp[threadIdx.x + i];
		__syncthreads();
	}
	__syncthreads();
}

// BUBBLE CLUSTER CONFIGURATION -------------------------------------------------------------------

void SetInitialConditionsAndParameters(double* h_StateX1, double* h_StateX2, double* h_Parameters, double* Frequencies, double* Amplitudes, double* BubbleSizes)
{
	srand (time(NULL));
	
	int Random;
	int Resolution = 100001;
	
	double P1;   // pressure amplitude1 [bar] (Shared among units, different among blocks; the first control parameter)
	double P2;   // frequency1          [kHz] (Shared among units, different among blocks; the second control parameter)
	double P3=0; // pressure amplitude2 [bar] (Zero; only for dual-frequency generalsation)
	double P4=0; // frequency2          [kHz] (Zero; only for dual-frequency generalsation)
	double P5=0; // phase shift         [-]   (Zero; only for dual-frequency generalsation)
	double P6;   // equilibrium radius  [mum] (Shared among blocks, different among units; NOT a control parameter, describes the bubble size distribution)
	
	double P7 = 1.0; // ambient pressure     [bar]
	double P9 = 1.4; // polytrophic exponent [-]
	
	double Pv  = 3.166775638952003e+03;
    double Rho = 9.970639504998557e+02;
    double ST  = 0.071977583160056;
    double Vis = 8.902125058209557e-04;
    double CL  = 1.497251785455527e+03;
	
	double Pinf;
	double PA1;
	double PA2;
	double RE;
	double f1;
	double f2;
	
	int GlobalIndex   = 0;
	int ParameterShift = CLUSTER_SIZE*FREQUENCY_RESOLUTION*AMPLITUDE_RESOLUTION*INITIAL_CONDITIONS;
	for (int i=0; i<FREQUENCY_RESOLUTION; i++) // Frequency [kHz]
	{
		for (int j=0; j<AMPLITUDE_RESOLUTION; j++) // Amplitude [bar]
		{
			for (int k=0; k<INITIAL_CONDITIONS; k++)
			{
				// BUBBLE SCOPE
				//cout << Amplitudes[j] << ", " << Frequencies[i] << endl;
				for (int b=0; b<CLUSTER_SIZE; b++) // equilibrium radius  [mum]
				{
					P1 = Amplitudes[j];  // pressure amplitude1 [bar]
					P2 = Frequencies[i]; // frequency1          [kHz]
					P6 = BubbleSizes[b]; // equilibrium radius  [mum]
					
					Random = (rand() % Resolution);
					h_StateX1[GlobalIndex] = RMIN + Random*(RMAX-RMIN)/Resolution;
					//h_StateX1[GlobalIndex] = 1.1;
					h_StateX2[GlobalIndex] = 0.0;
					
					// Dimensional physical parameters
					Pinf = P7 * 1e5;
					PA1  = P1 * 1e5;
					PA2  = P3 * 1e5;
					RE   = P6 / 1e6;
					
					f1   = 2.0*PI*(P2*1000.0);
					f2   = 2.0*PI*(P4*1000.0);
					
					h_Parameters[GlobalIndex +  0*ParameterShift] = (2.0*ST/RE + Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho;
					h_Parameters[GlobalIndex +  1*ParameterShift] = (1.0-3.0*P9) * (2*ST/RE + Pinf - Pv) * (2.0*PI/RE/f1) / CL/Rho;
					h_Parameters[GlobalIndex +  2*ParameterShift] = (Pinf - Pv) * pow(2.0*PI/RE/f1, 2.0) / Rho;
					h_Parameters[GlobalIndex +  3*ParameterShift] = (2.0*ST/RE/Rho) * pow(2.0*PI/RE/f1, 2.0);
					h_Parameters[GlobalIndex +  4*ParameterShift] = (4.0*Vis/Rho/pow(RE,2.0)) * (2.0*PI/f1);
					h_Parameters[GlobalIndex +  5*ParameterShift] = PA1 * pow(2.0*PI/RE/f1, 2.0) / Rho;
					h_Parameters[GlobalIndex +  6*ParameterShift] = PA2 * pow(2.0*PI/RE/f1, 2.0) / Rho;
					h_Parameters[GlobalIndex +  7*ParameterShift] = (RE*f1*PA1/Rho/CL) * pow(2.0*PI/RE/f1, 2.0);
					h_Parameters[GlobalIndex +  8*ParameterShift] = (RE*f2*PA2/Rho/CL) * pow(2.0*PI/RE/f1, 2.0);
					h_Parameters[GlobalIndex +  9*ParameterShift] = RE*f1/(2.0*PI)/CL;
					h_Parameters[GlobalIndex + 10*ParameterShift] = 3.0*P9;
					h_Parameters[GlobalIndex + 11*ParameterShift] = P4/P2;
					h_Parameters[GlobalIndex + 12*ParameterShift] = P5;
					
					h_Parameters[GlobalIndex + 13*ParameterShift] = 2.0*PI/f1;
					h_Parameters[GlobalIndex + 14*ParameterShift] = RE;
					
					h_Parameters[GlobalIndex + 15*ParameterShift] = P1;
					h_Parameters[GlobalIndex + 16*ParameterShift] = P2;
					h_Parameters[GlobalIndex + 17*ParameterShift] = P3;
					h_Parameters[GlobalIndex + 18*ParameterShift] = P4;
					h_Parameters[GlobalIndex + 19*ParameterShift] = P5;
					h_Parameters[GlobalIndex + 20*ParameterShift] = P6;
					
					GlobalIndex++;
				}
			}
		}
	}
}

void LoadBubbleSizes(double* BubbleSizes, string FileName, int N)
{
	ifstream DataFile;
	
	DataFile.open ( FileName.c_str() );
	DataFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	
	for ( int i=0; i<N; i++ )
		DataFile >> BubbleSizes[i];
	
	DataFile.close();
}

void LoadCouplingMatrix(double* CouplingMatrix, string FileName, int N)
{
	ifstream DataFile;
	
	DataFile.open ( FileName.c_str() );
	DataFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	
	int idx;
	for (int row=0; row<N; row++)
	{
		for (int col=0; col<N; col++)
		{
			idx = row + col*N;
			DataFile >> CouplingMatrix[idx];
		}
	}
	
	DataFile.close();
}

// DEVICE SELECTION HELPER FUNCTIONS --------------------------------------------------------------

void ListCUDADevices()
{
	int NumberOfDevices;
	cudaGetDeviceCount(&NumberOfDevices);
	for (int i = 0; i < NumberOfDevices; i++)
	{
		cudaDeviceProp CurrentDeviceProperties;
		cudaGetDeviceProperties(&CurrentDeviceProperties, i);
		std::cout << std::endl;
		std::cout << "Device number: " << i << std::endl;
		std::cout << "Device name:   " << CurrentDeviceProperties.name << std::endl;
		std::cout << "--------------------------------------" << std::endl;
		std::cout << "Total global memory:        " << CurrentDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
		std::cout << "Total shared memory:        " << CurrentDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
		std::cout << "Number of 32-bit registers: " << CurrentDeviceProperties.regsPerBlock << std::endl;
		std::cout << "Total constant memory:      " << CurrentDeviceProperties.totalConstMem / 1024 << " Kb" << std::endl;
		std::cout << std::endl;
		std::cout << "Number of multiprocessors:  " << CurrentDeviceProperties.multiProcessorCount << std::endl;
		std::cout << "Compute capability:         " << CurrentDeviceProperties.major << "." << CurrentDeviceProperties.minor << std::endl;
		std::cout << "Core clock rate:            " << CurrentDeviceProperties.clockRate / 1000 << " MHz" << std::endl;
		std::cout << "Memory clock rate:          " << CurrentDeviceProperties.memoryClockRate / 1000 << " MHz" << std::endl;
		std::cout << "Memory bus width:           " << CurrentDeviceProperties.memoryBusWidth  << " bits" << std::endl;
		std::cout << "Peak memory bandwidth:      " << 2.0*CurrentDeviceProperties.memoryClockRate*(CurrentDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
		std::cout << std::endl;
		std::cout << "Warp size:                  " << CurrentDeviceProperties.warpSize << std::endl;
		std::cout << "Max. warps per multiproc:   " << CurrentDeviceProperties.maxThreadsPerMultiProcessor / CurrentDeviceProperties.warpSize << std::endl;
		std::cout << "Max. threads per multiproc: " << CurrentDeviceProperties.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "Max. threads per block:     " << CurrentDeviceProperties.maxThreadsPerBlock << std::endl;
		std::cout << "Max. block dimensions:      " << CurrentDeviceProperties.maxThreadsDim[0] << " * " << CurrentDeviceProperties.maxThreadsDim[1] << " * " << CurrentDeviceProperties.maxThreadsDim[2] << std::endl;
		std::cout << "Max. grid dimensions:       " << CurrentDeviceProperties.maxGridSize[0] << " * " << CurrentDeviceProperties.maxGridSize[1] << " * " << CurrentDeviceProperties.maxGridSize[2] << std::endl;
		std::cout << std::endl;
		std::cout << "Concurrent memory copy:     " << CurrentDeviceProperties.deviceOverlap << std::endl;
		std::cout << "Execution multiple kernels: " << CurrentDeviceProperties.concurrentKernels << std::endl;
		std::cout << "ECC support turned on:      " << CurrentDeviceProperties.ECCEnabled << std::endl << std::endl;
	}
	std::cout << std::endl;
}

int SelectDeviceByClosestRevision(int MajorRevision, int MinorRevision)
{
	int SelectedDevice;
	
	cudaDeviceProp SelectedDeviceProperties;
	memset( &SelectedDeviceProperties, 0, sizeof(cudaDeviceProp) );
	
	SelectedDeviceProperties.major = MajorRevision;
	SelectedDeviceProperties.minor = MinorRevision;
		cudaChooseDevice( &SelectedDevice, &SelectedDeviceProperties );
	
	std::cout << "CUDA Device Number Closest to Revision " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << ": " << SelectedDevice << std::endl << std::endl << std::endl;
	
	return SelectedDevice;
}

void PrintPropertiesOfSpecificDevice(int SelectedDevice)
{
	cudaDeviceProp SelectedDeviceProperties;
	cudaGetDeviceProperties(&SelectedDeviceProperties, SelectedDevice);
	
	std::cout << "Selected device number: " << SelectedDevice << std::endl;
	std::cout << "Selected device name:   " << SelectedDeviceProperties.name << std::endl;
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "Total global memory:        " << SelectedDeviceProperties.totalGlobalMem / 1024 / 1024 << " Mb" << std::endl;
	std::cout << "Total shared memory:        " << SelectedDeviceProperties.sharedMemPerBlock / 1024 << " Kb" << std::endl;
	std::cout << "Number of 32-bit registers: " << SelectedDeviceProperties.regsPerBlock << std::endl;
	std::cout << "Total constant memory:      " << SelectedDeviceProperties.totalConstMem / 1024 << " Kb" << std::endl;
	std::cout << std::endl;
	std::cout << "Number of multiprocessors:  " << SelectedDeviceProperties.multiProcessorCount << std::endl;
	std::cout << "Compute capability:         " << SelectedDeviceProperties.major << "." << SelectedDeviceProperties.minor << std::endl;
	std::cout << "Core clock rate:            " << SelectedDeviceProperties.clockRate / 1000 << " MHz" << std::endl;
	std::cout << "Memory clock rate:          " << SelectedDeviceProperties.memoryClockRate / 1000 << " MHz" << std::endl;
	std::cout << "Memory bus width:           " << SelectedDeviceProperties.memoryBusWidth  << " bits" << std::endl;
	std::cout << "Peak memory bandwidth:      " << 2.0*SelectedDeviceProperties.memoryClockRate*(SelectedDeviceProperties.memoryBusWidth/8)/1.0e6 << " GB/s" << std::endl;
	std::cout << std::endl;
	std::cout << "Warp size:                  " << SelectedDeviceProperties.warpSize << std::endl;
	std::cout << "Max. warps per multiproc:   " << SelectedDeviceProperties.maxThreadsPerMultiProcessor / SelectedDeviceProperties.warpSize << std::endl;
	std::cout << "Max. threads per multiproc: " << SelectedDeviceProperties.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Max. threads per block:     " << SelectedDeviceProperties.maxThreadsPerBlock << std::endl;
	std::cout << "Max. block dimensions:      " << SelectedDeviceProperties.maxThreadsDim[0] << " * " << SelectedDeviceProperties.maxThreadsDim[1] << " * " << SelectedDeviceProperties.maxThreadsDim[2] << std::endl;
	std::cout << "Max. grid dimensions:       " << SelectedDeviceProperties.maxGridSize[0] << " * " << SelectedDeviceProperties.maxGridSize[1] << " * " << SelectedDeviceProperties.maxGridSize[2] << std::endl;
	std::cout << std::endl;
	std::cout << "Concurrent memory copy:     " << SelectedDeviceProperties.deviceOverlap << std::endl;
	std::cout << "Execution multiple kernels: " << SelectedDeviceProperties.concurrentKernels << std::endl;
	std::cout << "ECC support turned on:      " << SelectedDeviceProperties.ECCEnabled << std::endl << std::endl;
	
	std::cout << std::endl;
}

// DATA GENERATION HELPER FUNCTIONS ---------------------------------------------------------------

void Linspace(double* x, double B, double E, int N)
{
    double Increment;
	
	x[0]   = B;
	
	if ( N>1 )
	{
		x[N-1] = E;
		Increment = (E-B)/(N-1);
		
		for (int i=1; i<N-1; i++)
		{
			x[i] = B + i*Increment;
		}
	}
}

void Logspace(double* x, double B, double E, int N)
{
    x[0] = B; 
	
	if ( N>1 )
	{
		x[N-1] = E;
		double ExpB = log10(B);
		double ExpE = log10(E);
		double ExpIncr = (ExpE-ExpB)/(N-1);
		for (int i=1; i<N-1; i++)
		{
			x[i] = pow(10,ExpB + i*ExpIncr);
		}
	}
}