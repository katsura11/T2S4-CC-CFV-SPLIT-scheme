/* ****************************************************************************
  * File Name: ex1_T1S2_CFD.cu
  * INPUTS: config file
  * OUTPUTS: data_T1S2_**.txt
  * PURPOSE: This code is used to solve the 2D advection-diffusion equation with the 
  *          Temporal first order and Spatial second order conservative characteristic  
  *          finite volume scheme for Gauss hump problem.

  *          The equation is given by
  *          C_t + Vx C_x + Vy C_y = D (C_xx + C_yy) + f(x, y, t)
  *          where V = (Vx, Vy) = (-4y, 4x)
  *          C_Exact(x, y, t) = sigma / (sigma + 4*D*t) 
  *                             * exp(-((x - x0)^2 + (y - y0)^2) / (sigma + 4*D*t))
  *          C0(x, y) = exp(-((x - x0)^2 + (y - y0)^2) / sigma)
  *          x in [0, 1], y in [0, 1], t in [0, 1]
  ****************************************************************************** */

// check the error code
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

#include <iterator>
#define USE_GPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <ctime>
#include <cuda.h>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <fstream>
#include <device_atomic_functions.h>

#define x0 -0.35
#define y0  0.0
#define sigma 0.005
#define X_min -1.0
#define X_max  1.0
#define Y_min -1.0
#define Y_max  1.0

using namespace std;

/*********** Definition of variable types **********/
typedef int INT;
typedef double FLOAT;

/************************************ inline functions ************************************/
inline FLOAT Vx(FLOAT x, FLOAT y) { return -4 * y; }
inline FLOAT Vy(FLOAT x, FLOAT y) { return  4 * x; }
inline FLOAT C0(FLOAT x, FLOAT y) {
  return exp(-(pow((x - x0), 2) + pow((y - y0), 2)) / sigma);
}
inline FLOAT C_Exact(FLOAT x, FLOAT y, FLOAT t, FLOAT K) {
  FLOAT x_star =  x * cos(4 * t) + y * sin(4 * t);
  FLOAT y_star = -x * sin(4 * t) + y * cos(4 * t);
  return sigma / (sigma + 4 * K * t) *
         exp(-(pow((x_star - x0), 2) + pow((y_star - y0), 2)) / (sigma + 4 * K * t));
}

inline FLOAT sum(FLOAT *v_d_C, INT N_grid) {
  FLOAT sum = 0.0;
  for (int idx = 0; idx <= N_grid; idx++) {
    sum += v_d_C[idx];
  }
  return sum;
}

/*********************************** device functions ************************************/
__device__ FLOAT d_Vx(FLOAT x, FLOAT y) { return -4 * y; }
__device__ FLOAT d_Vy(FLOAT x, FLOAT y) { return  4 * x; }
__device__ FLOAT d_C0(FLOAT x, FLOAT y) {
  return exp(-(pow((x - x0), 2) + pow((y - y0), 2)) / sigma);
}
__device__ FLOAT d_C_Exact(FLOAT x, FLOAT y, FLOAT t, FLOAT K) {
    FLOAT x_star =  x * cos(4 * t) + y * sin(4 * t);
    FLOAT y_star = -x * sin(4 * t) + y * cos(4 * t);
    return sigma / (sigma + 4 * K * t) *
           exp(-(pow((x_star - x0), 2) + pow((y_star - y0), 2)) / (sigma + 4 * K * t));
}


void output_result(FLOAT *vec, INT t, INT Nx, INT Ny, FLOAT dt) {
  FILE *fp;
  char sfile[256];
  INT i_cell, j_cell;
  FLOAT x, y;
  FLOAT dx, dy;

  dx = (X_max - X_min) / Nx;
  dy = (Y_max - Y_min) / Ny;

  sprintf(sfile, "data_T1S2_%06d.txt", t);
  fp = fopen(sfile, "w");
  //fprintf(fp, "#time = %lf\n", (double)t * dt);
  //fprintf(fp, "#x y u\n");
  for (i_cell = 0; i_cell < Nx; i_cell++) {
    x = X_min + dx * (i_cell + 0.5);
    for (j_cell = 0; j_cell < Ny; j_cell++) {
      y = Y_min + dy * (j_cell + 0.5);
      fprintf(fp, "%.6lf %.6lf %.15lf\n", x, y, vec[i_cell * Ny + j_cell]);
      //fprintf(fp, "%.15e\n", vec[i_cell * Ny + j_cell]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return;
}


//******************************kenel functions********************************
__global__ void x_create_matrix_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC, FLOAT *v_d_b,
                                       FLOAT *v_d_C0, FLOAT *v_d_C0_bar, FLOAT *v_d_f1, FLOAT dx, 
                                       FLOAT dy, FLOAT t, FLOAT dt, INT Nx, INT Ny, INT N_grid, 
                                       FLOAT Kx, FLOAT dt_mul_dx_square_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x, y, x_bar, x_m, ksi;
  int i, j, m;

  while (ind < N_grid) {
    i = ind / Ny;
    j = ind % Ny;

    x = X_min + dx * (i + 0.5);
    y = Y_min + dy * (j + 0.5);
    x_bar = x - d_Vx(x, y) * dt;
    m = floor((x_bar - X_min) / dx);
    x_m = X_min + dx * (m + 0.5);
    ksi = (x_bar - x_m) / dx;

    //******************** calculate U_bar in x-direction ************************
    if (m > 0 && m < Nx - 1) {
      v_d_C0_bar[ind] = 0.5 * v_d_C0[(m - 1) * Ny + j] * (ksi - 1) * ksi 
                        -       v_d_C0[m * Ny + j] * (ksi - 1) * (ksi + 1)
                        + 0.5 * v_d_C0[(m + 1) * Ny + j] * (ksi + 1) * ksi;
    }
    else v_d_C0_bar[ind] = 0;

    //*********************** assemble matrix elements ****************************
    v_d_coA[ind] = -Kx * dt_mul_dx_square_inverse;			       // lower-diagonal
    v_d_coB[ind] = 1.0 + 2.0 * Kx * dt_mul_dx_square_inverse; // main-diagonal
    v_d_coC[ind] = -Kx * dt_mul_dx_square_inverse;			     // upper-diagonal
    v_d_b[ind] = v_d_C0_bar[ind];
		
	ind += gridDim.x * blockDim.x;
	}
	return;
}

__global__ void y_create_matrix_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC, FLOAT *v_d_b,
                                        FLOAT *v_d_C_xphf, FLOAT *v_d_C_xphf_bar, FLOAT *v_d_f2, 
                                        FLOAT dx, FLOAT dy, FLOAT t, FLOAT dt, INT Nx, INT Ny, 
                                        INT N_grid, FLOAT Ky, FLOAT dt_mul_dy_square_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x, y, y_bar, y_m, ksi;
  int i, j, m;

  while (ind < N_grid) {
    i = ind / Ny;
    j = ind % Ny;

    x = X_min + dx * (i + 0.5);
    y = Y_min + dy * (j + 0.5);
    y_bar = y - d_Vy(x, y) * dt;
    m = floor((y_bar - Y_min) / dy);
    y_m = Y_min + dy * (m + 0.5);
    ksi = (y_bar - y_m) / dy;

    //******************** calculate U_bar in y-direction ************************
    if (m > 0 && m < Ny - 1) {
      v_d_C_xphf_bar[ind] = 0.5 * v_d_C_xphf[i * Ny + m - 1] * (ksi - 1) * ksi -
                            v_d_C_xphf[i * Ny + m] * (ksi - 1) * (ksi + 1) +
                            0.5 * v_d_C_xphf[i * Ny + m + 1] * (ksi + 1) * ksi;
    }
    else v_d_C_xphf_bar[ind] = 0;

    //*********************** assemble matrix elements ****************************
    v_d_coA[ind] = -Ky * dt_mul_dy_square_inverse;		     // lower-diagonal
    v_d_coB[ind] = 1 + 2 * Ky * dt_mul_dy_square_inverse; // main-diagonal
    v_d_coC[ind] = -Ky * dt_mul_dy_square_inverse;		   // upper-diagonal																	 
    v_d_b[ind] = v_d_C_xphf_bar[ind];

    ind += gridDim.x * blockDim.x;
	}
	return;
}

__global__ void x_Thomas_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC,
                                FLOAT *v_d_b, FLOAT *v_d_C_xphf, FLOAT *p,
                                FLOAT *q, INT N_grid, INT Nx, INT Ny) 
{
  INT ins = blockDim.x * blockIdx.x + threadIdx.x;
  INT i_cell, j_cell;
  FLOAT denom;
  while (ins < Ny) {
    j_cell = ins;

    p[j_cell] = v_d_coC[j_cell] / v_d_coB[j_cell];
    q[j_cell] = (v_d_b[j_cell]) / v_d_coB[j_cell];

    for (i_cell = 1; i_cell < Nx; i_cell++) {
      denom = 1.0f / (v_d_coB[i_cell * Ny + j_cell] - p[(i_cell - 1) * Ny + j_cell] * v_d_coA[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_d_coC[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_d_b[i_cell * Ny + j_cell] - q[(i_cell - 1) * Ny + j_cell] * v_d_coA[i_cell * Ny + j_cell]) * denom;
    }

    v_d_C_xphf[(Nx - 1) * Ny + j_cell] = q[(Nx - 1) * Ny + j_cell];
    for (int i_cell = Nx-2; i_cell >= 0; i_cell--) {
      v_d_C_xphf[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] - p[i_cell * Ny + j_cell] * v_d_C_xphf[(i_cell + 1) * Ny + j_cell];
    }    
    ins += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void y_Thomas_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC,
                                FLOAT *v_d_b, FLOAT *v_d_Cp1, FLOAT *p,
                                FLOAT *q, INT N_grid, INT Nx, INT Ny) 
{
  int ins = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT denom;

  while (ins < Nx) {
    i_cell = ins ;

    // q[0]=q[0]/b[0] ; d[0]=d[0]/b[0];
    p[i_cell * Ny] = v_d_coC[i_cell * Ny] / v_d_coB[i_cell * Ny];
    q[i_cell * Ny] = (v_d_b[i_cell * Ny]) / v_d_coB[i_cell * Ny];

    for (int j_cell = 1; j_cell < Ny; j_cell++) {
      denom = 1.0f / (v_d_coB[i_cell * Ny + j_cell] - p[i_cell * Ny + j_cell - 1] * v_d_coA[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_d_coC[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_d_b[i_cell * Ny + j_cell] - q[i_cell * Ny + j_cell - 1] * v_d_coA[i_cell * Ny + j_cell]) * denom;
    }

    v_d_Cp1[i_cell * Ny + Ny-1] = q[i_cell * Ny + Ny-1];
    for (int j_cell = Ny-2; j_cell >= 0; j_cell--) {
      v_d_Cp1[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] - p[i_cell * Ny + j_cell] * v_d_Cp1[i_cell * Ny + (j_cell + 1)];
    }  
    ins += gridDim.x * blockDim.x;
  }
  return;
}

//*********************************
// Main Code
//*********************************

int main(int argc, char *argv[]) {
  // Start of the time controller
  clock_t start = clock();
  //*************************************
  // Initialization of host variables
  //*************************************
  FLOAT Kx, Ky;
  FLOAT T_min, T_max; 
  FLOAT dx, dy, dt;
  FLOAT x, y, t0;
  INT i_cell, j_cell, itime, tid; 
  INT Nx, Ny, Nt, T_span, N_grid, N_points;
  FLOAT E_2, E_Inf, Err, Cmassdif, cmass_initial, cmass_end;
  FLOAT *v_h_f, *v_h_C0, *v_h_C1, *v_h_C_Exact, *v_d_C_Exact;
  FLOAT *v_d_C0, *v_d_C0_bar, *v_d_C_xphf, *v_d_C_xphf_bar, *v_d_Cp1; 
  FLOAT *v_d_coA, *v_d_coB, *v_d_coC, *v_d_b, *p, *q;
  FLOAT *v_d_f0, *v_d_f1;


  if (argc < 2) {
    cout << "please input config file name" << endl;
  }

  //*************************************************
  // read  parameters from config file
  //*************************************************
  ifstream configFile;
  configFile.open(argv[1]);
  string strLine;
  string strKey, strValue;
  size_t pos;
  if (configFile.is_open()) {
    cout << "open config file ok" << endl;
    while (!configFile.eof()) {
      getline(configFile, strLine);
      pos = strLine.find(':');
      strKey = strLine.substr(0, pos);
      strValue = strLine.substr(pos + 1);
      if (strKey.compare("T_min") == 0) {
        sscanf(strValue.c_str(), "%lf", &T_min);
      }
      if (strKey.compare("T_max") == 0) {
        sscanf(strValue.c_str(), "%lf", &T_max);
      }
      if (strKey.compare("N") == 0) {
        sscanf(strValue.c_str(), "%d", &Nx);
      }
      if (strKey.compare("N") == 0) {
        sscanf(strValue.c_str(), "%d", &Ny);
      }
      if (strKey.compare("Nt") == 0) {
        sscanf(strValue.c_str(), "%d", &Nt);
      }
      if (strKey.compare("T_span") == 0) {
        sscanf(strValue.c_str(), "%d", &T_span);
      }
      if (strKey.compare("K") == 0) {
        sscanf(strValue.c_str(), "%lf", &Kx);
      }
      if (strKey.compare("K") == 0) {
        sscanf(strValue.c_str(), "%lf", &Ky);
      }
    }
  } else {
    cout << "Cannot open config file!" << endl;
    return 1;
  }
  configFile.close();//	fclose(setup_file);

  // Calculate delta_x, delta_y, delta_t, and the problem size N_grid after
  // discretization
  dx = (X_max - X_min) / Nx;
  dy = (Y_max - Y_min) / Ny;
  dt = (T_max - T_min) / Nt;

  N_grid = Nx * Ny;
  N_points = (Nx + 1) * (Ny + 1);


  FLOAT dt_half = 0.5 * dt;
  FLOAT dt_mul_dx_square_inverse = dt / (dx * dx);
  FLOAT dt_mul_dy_square_inverse = dt / (dy * dy);
  

  // GPU related variables and parameters
  int numBlocks;       // Number of blocks
  int threadsPerBlock; // Number of threads
  int maxThreadsPerBlock;
  cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	//maxThreadsPerBlock = prop.maxThreadsPerBlock;
  maxThreadsPerBlock = 256;
	if (N_points < maxThreadsPerBlock)
	{
		threadsPerBlock = N_points;
		numBlocks = 1;
	}
	else
	{
		threadsPerBlock = maxThreadsPerBlock;
		numBlocks = (N_points + maxThreadsPerBlock - 1) / maxThreadsPerBlock;
	}

  // Allocate host memory for the location vector
  v_h_f = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_C0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_C1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_C_Exact = (FLOAT *)malloc(N_grid * sizeof(FLOAT));

  // Create device vector to store the solution at each time step
  cudaMalloc((void **)&v_d_C0, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cp1, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C0_bar, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C_xphf, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C_xphf_bar, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C_Exact, N_grid * sizeof(FLOAT));

  // Create device vector to store matrix elements
  cudaMalloc((void **)&p, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&q, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_b, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coA, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coB, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coC, N_grid * sizeof(FLOAT));

  // Create device vector to store source term
  cudaMalloc((void **)&v_d_f0, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_f1, N_grid * sizeof(FLOAT));

// Calculte Initial condition and Exact solution on CPU
  cmass_initial = 0.0;
  for (i_cell = 0; i_cell < Nx; i_cell++){
    x = X_min + (i_cell + 0.5) * dx;
      for (j_cell = 0; j_cell < Ny; j_cell++){
          y = Y_min + (j_cell + 0.5) * dy;
          tid = i_cell * Ny + j_cell;       
          v_h_C0[tid] = C0(x, y);
          v_h_C_Exact[tid] = C_Exact(x, y, T_min, Kx);
          cmass_initial = cmass_initial + v_h_C0[tid] * dx * dy;
      }
  } 
  cudaMemcpy(v_d_C0, v_h_C0, N_grid * sizeof(FLOAT), cudaMemcpyHostToDevice);
  output_result(v_h_C0, 0, Nx, Ny, dt);

  FLOAT sumf = 0.0;
  for (itime = 1; itime <= Nt; itime++){
    //cout << "itime = " << itime << endl;
    t0 = (itime - 1) * dt;
    //****************************** step 1************************************
    x_create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_C0, v_d_C0_bar, v_d_f0, 
    dx, dy, t0, dt, Nx, Ny, N_grid, Kx, dt_mul_dx_square_inverse);
    cudaDeviceSynchronize();

    x_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_C_xphf, p, q, N_grid, Nx, Ny);
    cudaDeviceSynchronize();

    //****************************** step 2 ***********************************
    y_create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_C_xphf, v_d_C_xphf_bar, v_d_f1,
    dx, dy, t0, dt, Nx, Ny, N_grid, Ky,dt_mul_dx_square_inverse);
    cudaDeviceSynchronize();

    y_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_Cp1, p, q, N_grid, Nx, Ny);
    cudaDeviceSynchronize();

    //************************* Update the solution *****************************
    cudaMemcpy(v_d_C0, v_d_Cp1, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
    
    // 输出数据
    if (itime % T_span == 0) {
      cudaMemcpy(v_h_C1, v_d_C0, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
      output_result(v_h_C1, itime, Nx, Ny, dt);
    }  
  
  }
  cudaMemcpy(v_h_C1, v_d_C0, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);

 // calculate error
  E_2 = 0.0;
  E_Inf = 0.0;
  Err = 0.0;
  Cmassdif = 0.0;
  cmass_end = 0.0;
  cmass_initial = 0.0;
  
  for (tid = 0; tid < N_grid; tid++) {
    Err = v_h_C1[tid] - v_h_C_Exact[tid];
    E_2 += Err * Err;
    E_Inf = fmax(E_Inf, fabs(Err));
    cmass_end = cmass_end + v_h_C1[tid] * dx * dy;
    cmass_initial = cmass_initial + v_h_C0[tid] * dx * dy;
  }

  Cmassdif = cmass_initial - cmass_end;
  //E_2 = sqrt(E_2 * dx *dy);
  E_2 = sqrt(E_2) * dx;
  
  cout << "K:" << Kx << fixed << setprecision(16) << endl;
  cout << "dx:" << dx << endl;
  cout << "dy:" << dy << endl;
  cout << "dt:" << dt << endl;
  cout << "Nx:" << Nx << endl;
  cout << "Ny:" << Ny << endl;
  cout << "Nt:" << Nt << endl;
  printf("E2:%.4e\n", E_2); 
  printf("\n");
  printf("EInf:%.4e\n", E_Inf);
  printf("\n");
  printf("Cmassdif:%.15e\n", Cmassdif);
  //printf("cmass_end:%.15e\n", cmass_end);
  //printf("cmass_initial:%.15e\n", cmass_initial);
  
  
  clock_t end = clock();
  double time = (end - start) / (double)CLOCKS_PER_SEC;
  cout << "time:" << time << endl;


  cudaFree(v_d_C0);
  cudaFree(v_d_Cp1);
  cudaFree(v_d_b);
  cudaFree(v_d_coA);
  cudaFree(v_d_coB);
  cudaFree(v_d_coC);
  cudaFree(v_d_f0);
  cudaFree(v_d_f1);
  cudaFree(v_d_C0_bar);
  cudaFree(v_d_C_xphf);
  cudaFree(v_d_C_xphf_bar);
  cudaFree(p);
  cudaFree(q);

  return 0;
}