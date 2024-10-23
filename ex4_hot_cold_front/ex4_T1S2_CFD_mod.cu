#include <iterator>
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

#define X_min -4.0
#define X_max  4.0
#define Y_min -4.0
#define Y_max  4.0

using namespace std;

/*********** Definition of variable types **********/
typedef int INT;
typedef double FLOAT;

/************************************ inline functions ************************************/
inline FLOAT C0(FLOAT x, FLOAT y) {
  return -tanh(y/2);
}
inline FLOAT C_Exact(FLOAT x, FLOAT y, FLOAT t) {
  FLOAT r = sqrt(x*x + y*y);
  FLOAT f_t = tanh(r)/(cosh(r) * cosh(r));
  return -tanh(y/2*cos(f_t*t/(0.385*r)) - x/2*sin(f_t*t/(0.385*r)));
}

/************************************ device functions ************************************/
__device__ FLOAT d_Vx(FLOAT x, FLOAT y) { 
  FLOAT r = sqrt(x*x + y*y);
  FLOAT f_t = tanh(r)/(cosh(r) * cosh(r));
  if (r == 0) return 0;
  else return -(y/r) * (f_t/0.385); 
}
__device__ FLOAT d_Vy(FLOAT x, FLOAT y) { 
  FLOAT r = sqrt(x*x + y*y);
  FLOAT f_t = tanh(r)/(cosh(r) * cosh(r));
  if (r == 0) return 0;
  else return  (x/r) * (f_t/0.385);
}
__device__ FLOAT d_C0(FLOAT x, FLOAT y) {
  return -tanh(y/2);
}
__device__ FLOAT d_C_Exact(FLOAT x, FLOAT y, FLOAT t) {
  FLOAT r = sqrt(x*x + y*y);
  FLOAT f_t = tanh(r)/(cosh(r) * cosh(r));
  return -tanh(y/2*cos(f_t*t/(0.385*r)) - x/2*sin(f_t*t/(0.385*r)));
}
__device__ FLOAT d_fIntC(FLOAT vC6, FLOAT vDltc, FLOAT vCL, FLOAT xi) {
  return -1.0 / 3 * vC6 * pow(xi, 3) + 1.0 / 2 * (vDltc + vC6) * pow(xi, 2) + vCL * xi;
}
__device__ FLOAT d_x_sum(FLOAT *v_C0, INT iL, INT iR, INT j, INT Ny) {
  FLOAT sum = 0.0;
  for (int i_cell = iL; i_cell <= iR; i_cell++) {
    sum += v_C0[i_cell * Ny + j];
  }
  return sum;
}
__device__ FLOAT d_y_sum(FLOAT *v_C0, INT jL, INT jR, INT i, INT Ny) {
  FLOAT sum = 0;
  for (int j_cell = jL; j_cell <= jR; j_cell++) {
    sum += v_C0[i* Ny + j_cell];
  }
  return sum;
}

// Output the result to a file
void output_result(FLOAT *vec, INT t, INT Nx, INT Ny, FLOAT dt) {
    FILE *fp;
    char sfile[256];
    int i_cell, j_cell;
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
          fprintf(fp, "%.6lf %.6lf %.4e\n", x, y, vec[i_cell * Ny + j_cell]);
          //fprintf(fp, "%.15e\n", vec[i_cell * Ny + j_cell]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return;
}

//****************************** kernel function **********************
__global__ void x_create_matrix_on_gpu(FLOAT *v_d_C0, FLOAT *v_d_C0_bar, FLOAT dx, FLOAT dy, 
                                        FLOAT t, FLOAT dt, INT Nx, INT Ny, INT N_grid, 
                                        FLOAT dt_mul_dx_square_inverse)
{
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	FLOAT x, y, x_bar, x_m, ksi;
	int i, j, m;
	while (ind < N_grid)
	{
		i = ind / Ny ;
		j = ind % Ny ;
		x = X_min + dx * (i + 0.5);
		y = Y_min + dy * (j + 0.5);
		x_bar = x - d_Vx(x, y) * dt;
		m = floor((x_bar - X_min) / dx);
		x_m = X_min + dx * (m + 0.5);
		ksi = (x_bar - x_m)/dx;
		
		//*********************** calculate U_bar **************************
		if (m > 0 && m < Nx - 1)
		{
			v_d_C0_bar[ind] = 0.5 * v_d_C0[(m - 1) * Ny + j] * (ksi - 1) * ksi 
                            - v_d_C0[m * Ny + j] * (ksi - 1) * (ksi + 1) 
			                + 0.5 * v_d_C0[(m + 1) * Ny + j] * (ksi + 1) * ksi ;
		}
    else v_d_C0_bar[ind] = d_C_Exact(x, y, t-dt);

		ind += gridDim.x * blockDim.x;
	}
	return;
}

__global__ void y_create_matrix_on_gpu(FLOAT *v_d_C_xphf, FLOAT *v_d_C_xphf_bar, 
                                      FLOAT dx, FLOAT dy, FLOAT t, FLOAT dt, INT Nx, INT Ny, 
                                      INT N_grid, FLOAT dt_mul_dy_square_inverse)
{
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	FLOAT x, y, y_bar, y_m, ksi;
	int i, j, m;
	while (ind < N_grid)
	{
		i = ind / Ny ;
		j = ind % Ny ;
		x = X_min + dx * (i + 0.5);
		y = Y_min + dy * (j + 0.5);
		y_bar = y - d_Vy(x, y) * dt;
		m = floor((y_bar - Y_min) / dy);
		y_m = Y_min + dy * (m + 0.5);
		ksi = (y_bar - y_m)/dy;

		//************************* calculate U_bar ************************
		if (m > 0 && m < Ny - 1)
		{
			v_d_C_xphf_bar[ind] = 0.5 * v_d_C_xphf[i * Ny + m - 1] * (ksi - 1) * ksi 
                                - v_d_C_xphf[i * Ny + m] * (ksi - 1) * (ksi + 1)
			                 	+ 0.5 * v_d_C_xphf[i * Ny + m + 1] * (ksi + 1) * ksi;
		}
    else v_d_C_xphf_bar[ind] = d_C_Exact(x, y, t-dt);

		ind += gridDim.x * blockDim.x;
	}
	return;
}

//*********************************
// Main Code
//*********************************
int main(int argc, char *argv[]) {
    // Start of the time controller
    clock_t start = clock();

    //**************** initialize variables ******************
    FLOAT T_min, T_max; 
    FLOAT dx, dy, dt;
    FLOAT x, y, t0, t1;
    INT i_cell, j_cell, itime, tid; // 循环变量
    INT Nx, Ny, Nt, T_span, N_grid, N_points;
    FLOAT E_2, E_Inf, Err, Cmassdif, cmass_initial, cmass_end;
    FLOAT *v_h_C0, *v_h_C1, *v_h_C_Exact;
    FLOAT *v_d_C0, *v_d_C0_bar, *v_d_C_xphf, *v_d_C_xphf_bar, *v_d_Cp1; 

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
        }
    } else {
        cout << "Cannot open config file!" << endl;
        return 1;
    }
    configFile.close();
    //fclose(setup_file);

    // Calculate delta_x, delta_y, delta_t, and the problem size 
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
    v_h_C0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
    v_h_C1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
    v_h_C_Exact = (FLOAT *)malloc(N_grid * sizeof(FLOAT));

    // Create device vector to store the solution 
    cudaMalloc((void **)&v_d_C0, N_grid * sizeof(FLOAT));
    cudaMalloc((void **)&v_d_Cp1, N_grid * sizeof(FLOAT));
    cudaMalloc((void **)&v_d_C0_bar, N_grid * sizeof(FLOAT));
    cudaMalloc((void **)&v_d_C_xphf, N_grid * sizeof(FLOAT));
    cudaMalloc((void **)&v_d_C_xphf_bar, N_grid * sizeof(FLOAT));

    // Calculte Initial condition and Exact solution on CPU
    clock_t t2 = clock();
    cmass_initial = 0.0;
    for (i_cell = 0; i_cell < Nx; i_cell++){
        x = X_min + (i_cell + 0.5) * dx;
        for (j_cell = 0; j_cell < Ny; j_cell++){
            y = Y_min + (j_cell + 0.5) * dy;
            tid = i_cell * Ny + j_cell;       
            v_h_C0[tid] = C_Exact(x, y, 0);
            v_h_C_Exact[tid] = C_Exact(x, y, T_max);
            cmass_initial = cmass_initial + v_h_C0[tid] * dx * dy;
        }
    }
    cudaMemcpy(v_d_C0, v_h_C0, N_grid * sizeof(FLOAT), cudaMemcpyHostToDevice);
    output_result(v_h_C0, 0, Nx, Ny, dt);

    // Time marching loop
    for (itime = 1; itime <= Nt; itime++){
        //cout << "itime = " << itime << endl;
        t1 = itime * dt;
        //**************************** step 1**********************************
        x_create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
        (v_d_C0, v_d_C0_bar, dx, dy, t1, dt, Nx, Ny, N_grid, dt_mul_dx_square_inverse);
        cudaDeviceSynchronize();
        v_d_C_xphf = v_d_C0_bar;

        //***************************** step 2 *********************************
        y_create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
        (v_d_C_xphf, v_d_C_xphf_bar, dx, dy, t1, dt, Nx, Ny, N_grid, dt_mul_dx_square_inverse);
        cudaDeviceSynchronize();
        v_d_C0 = v_d_C_xphf_bar;

        // Output the result 
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
    
    for (tid = 0; tid < N_grid; tid++) {
        Err = v_h_C1[tid] - v_h_C_Exact[tid];
        E_2 += Err * Err;
        E_Inf = fmax(E_Inf, fabs(Err));
        cmass_end = cmass_end + v_h_C1[tid] * dx * dy;
    }

    Cmassdif = fabs(cmass_initial - cmass_end);
    //E_2 = sqrt(E_2 * dx *dy);
    E_2 = sqrt(E_2) * dx;
    
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
    printf("cmass_end:%.15e\n", cmass_end);
    printf("cmass_initial:%.15e\n", cmass_initial);
        
    clock_t end = clock();
    double time = (end - start) / (double)CLOCKS_PER_SEC;
    cout << "time:" << time << endl;

    cudaFree(v_d_C0);
    cudaFree(v_d_Cp1);
    cudaFree(v_d_C0_bar);
    cudaFree(v_d_C_xphf);
    cudaFree(v_d_C_xphf_bar);

    return 0;
}