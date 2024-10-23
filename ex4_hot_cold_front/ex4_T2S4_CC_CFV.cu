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
inline FLOAT sum(FLOAT *v_d_C, INT N_grid) {
  FLOAT sum = 0.0;
  for (int idx = 0; idx <= N_grid; idx++) {
    sum += v_d_C[idx];
  }
  return sum;
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
    if (r == 0) return 0;
    else return -tanh(y/2*cos(f_t*t/(0.385*r)) - x/2*sin(f_t*t/(0.385*r)));
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

void output_result(FLOAT *vec, INT t, INT Nx, INT Ny, FLOAT dt) {
  FILE *fp;
  char sfile[256];
  int i_cell, j_cell;
  FLOAT x, y;
  FLOAT dx, dy;

  dx = (X_max - X_min) / Nx;
  dy = (Y_max - Y_min) / Ny;

  sprintf(sfile, "data_T2S4_%06d.txt", t);
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

/************************************ kernel functions ************************************/
// calculate int_points location
__global__ void Get_Euler_points_on_gpu(FLOAT *v_d_x, FLOAT *v_d_y, INT Nx, INT Ny, 
                                        INT N_grid, FLOAT dx, FLOAT dy) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  INT i_cell, j_cell;

  while (tid < N_grid) {

    i_cell = tid / Ny; // i_cell = 0, 1, ..., Nx-1
    j_cell = tid % Ny; // i_cell = 0, 1, ..., Ny-1
    
    // calculate Eulerian points
    v_d_x[tid] = X_min + (i_cell + 0.5) * dx;
    v_d_y[tid] = Y_min + (j_cell + 0.5) * dy;

    tid += gridDim.x * blockDim.x;
  }
  return;
}

// calculate half_points location
__global__ void Get_Lagrange_points_on_gpu(FLOAT *v_d_xhf, FLOAT *v_d_yhf, FLOAT *v_d_xhf_bar1, 
                                          FLOAT *v_d_yhf_bar2, FLOAT *v_d_xhf_bar3, INT Nx, INT Ny, 
                                          INT N_points, FLOAT dx, FLOAT dy, FLOAT dt) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x_star1, y_star2, x_star3;
  INT i_point, j_point;

  while (tid < N_points) {

    i_point = tid / (Ny+1); // i_point = 0, 1, ..., Nx
    j_point = tid % (Ny+1); // i_point = 0, 1, ..., Ny

    // calculate Eulerian points
    v_d_xhf[tid] = X_min + dx * i_point;
    v_d_yhf[tid] = Y_min + dy * j_point;
    
    // calculate Lagrangian points
    /**************************************** step 1 *******************************************/
    x_star1 = v_d_xhf[tid] - 0.5 * dt * d_Vx(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_xhf_bar1[tid] = v_d_xhf[tid] - 0.25 * dt * (d_Vx(v_d_xhf[tid], v_d_yhf[tid]) + d_Vx(x_star1, v_d_yhf[tid]));

    /**************************************** step 2 *******************************************/
    y_star2 = v_d_yhf[tid] -       dt * d_Vy(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_yhf_bar2[tid] = v_d_yhf[tid] -  0.5 * dt * (d_Vy(v_d_xhf[tid], v_d_yhf[tid]) + d_Vy(v_d_xhf[tid], y_star2));    
    
    /**************************************** step 3 *******************************************/
    x_star3 = v_d_xhf[tid] - 0.5 * dt * d_Vx(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_xhf_bar3[tid] = v_d_xhf[tid] - 0.25 * dt * (d_Vx(v_d_xhf[tid], v_d_yhf[tid]) + d_Vx(x_star3, v_d_yhf[tid]));

    tid += gridDim.x * blockDim.x;
  }
  return;
}


// Calculate intersection points in the x-direction
__global__ void Intersection_point_vet_x(FLOAT *v_d_xhf_bar1, FLOAT *v_d_yhf,
                                         FLOAT *v_d_y, FLOAT *v_d_x_inter1,
                                         INT Nx, INT Ny, INT N_points_inter1) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < N_points_inter1) {
    INT j_cell = tid % Ny;   // j_cell = 0, 1, ..., Ny-1
    INT i_point = tid / Ny;  // i_point = 0, 1, ..., Nx
    INT ind = i_point * (Ny + 1) + j_cell;

    // Calculate intersection points
    FLOAT tmp = 1.0 / (v_d_yhf[ind] - v_d_yhf[ind + 1]);
    v_d_x_inter1[tid] = v_d_xhf_bar1[ind + 1] 
                        + (v_d_xhf_bar1[ind] -v_d_xhf_bar1[ind + 1]) * tmp
                        * (v_d_y[j_cell] - v_d_yhf[ind + 1]);

    tid += gridDim.x * blockDim.x;
  }
  return;
}

// Calculate intersection points in the y-direction
__global__ void Intersection_point_hori_y(FLOAT *v_d_xhf, FLOAT *v_d_yhf_bar2,
                                          FLOAT *v_d_x, FLOAT *v_d_y_inter2,
                                          INT Nx, INT Ny, INT N_points_inter2) 
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < N_points_inter2) {
    INT i_cell = tid / (Ny + 1);   // i_cell = 0, 1, ..., Nx-1
    INT j_point = tid % (Ny + 1);  // j_point = 0, 1, ..., Ny
    
    INT ind = i_cell * (Ny + 1) + j_point;
    INT ind_next = (i_cell + 1) * (Ny + 1) + j_point;

    // Calculate intermediate x coordinate
    FLOAT x_int = 0.5 * (v_d_xhf[ind] + v_d_xhf[ind_next]);

    // Calculate intersection points
    FLOAT tmp = 1.0 / (v_d_xhf[ind] - v_d_xhf[ind_next]);
    v_d_y_inter2[tid] = v_d_yhf_bar2[ind_next] 
                      + (v_d_yhf_bar2[ind] - v_d_yhf_bar2[ind_next]) * tmp 
                      * (x_int - v_d_xhf[ind_next]);

    tid += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void x_Remap_on_gpu_part1(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6, FLOAT *v_d_Dltc, FLOAT *v_d_x, 
                                      FLOAT *v_d_y, FLOAT *v_d_C, INT N_grid, INT Nx, INT Ny, FLOAT t) 
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT bL, bR;
  int i_cell, j_cell;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;

  FLOAT siv_twelfth = 7.0 / 12.0;
  FLOAT one_twelfth = 1.0 / 12.0;

  while (ind < N_grid) {

    i_cell = ind / Ny;
    j_cell = ind % Ny;

    // ************************ calculate ghost cell values ************************
    bL = d_C_Exact(v_d_x[0], v_d_y[ind], t);
    C0m1 = 1.0 / 3.0 * (-13 * v_d_C[j_cell] + 5 * v_d_C[Ny + j_cell] - v_d_C[2*Ny + j_cell] + 12 * bL);
    C0m2 = 1.0 / 3.0 * (-70 * v_d_C[j_cell] + 32 * v_d_C[Ny + j_cell] - 7 * v_d_C[2*Ny + j_cell] + 48 * bL);

    bR = d_C_Exact(v_d_x[N_grid - 1], v_d_y[ind], t);
    Cn = 1.0 / 3.0 * (-13 * v_d_C[(Nx-1)*Ny + j_cell] + 5 * v_d_C[(Nx-2)*Ny + j_cell] - v_d_C[(Nx-3)*Ny + j_cell] + 12 * bR);
    Cnp1 = 1.0 / 3.0 * (-70 * v_d_C[(Nx-1)*Ny + j_cell] + 32 * v_d_C[(Nx-2)*Ny + j_cell] - 7 * v_d_C[(Nx-3)*Ny + j_cell] + 48 * bR);

    // ************************ set values for Ci, Cip1, Cip2, Cim1, Cim2 ************************
    Ci = v_d_C[i_cell * Ny + j_cell];
    Cip1 = i_cell == Nx-1 ? Cn : v_d_C[(i_cell + 1) * Ny + j_cell];
    Cip2 = i_cell >= Nx-2 ? ((i_cell == Nx-2) ? Cn : Cnp1) : v_d_C[(i_cell + 2) * Ny + j_cell];
    Cim1 = i_cell == 0 ? C0m1 : v_d_C[(i_cell - 1) * Ny + j_cell];
    Cim2 = i_cell <= 1 ? ((i_cell == 1) ? C0m1 : C0m2) : v_d_C[(i_cell - 2) * Ny + j_cell];
    
    //************************ calculate CL, CR *************************** 
    v_d_CL[ind] = siv_twelfth * (Ci + Cim1) - one_twelfth * (Cip1 + Cim2);
    v_d_CR[ind] = siv_twelfth * (Cip1 + Ci) - one_twelfth * (Cip2 + Cim1);

    //**************** calculate Dltc, C6 ***************
    v_d_Dltc[ind] = v_d_CR[ind] - v_d_CL[ind];
    v_d_C6[ind] = 6.0 * (v_d_C[ind] - 0.5 * (v_d_CL[ind] + v_d_CR[ind]));
       
    ind += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void x_Remap_on_gpu_part2(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6, 
                                     FLOAT *v_d_Dltc, FLOAT *v_d_C, FLOAT *v_d_CRemap,
                                     FLOAT *v_d_xhf, FLOAT *v_d_y, FLOAT *v_d_x_inter1, 
                                     INT N_grid, INT Nx, INT Ny, FLOAT dx, FLOAT t) 
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0;

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;

    // If xbL and xbR are outside the boundary or equal, then IntC = 0
    // Otherwise, calculate the integral value
    FLOAT intxbL = fmax(v_d_x_inter1[ind], X_min + 1e-15);
    FLOAT intxbR = fmin(v_d_x_inter1[ind + Ny], X_max - 1e-15);
    INT iL = floor((intxbL - X_min) / dx);
    INT iR = floor((intxbR - X_min) / dx);
    FLOAT xL_ksi = (intxbL - v_d_xhf[iL * (Ny + 1) + j_cell]) / dx;
    FLOAT xR_ksi = (intxbR - v_d_xhf[iR * (Ny + 1) + j_cell]) / dx;

    //************************* Remap of C****************************
    INT id_R = iR * Ny + j_cell;
    INT id_L = iL * Ny + j_cell;

    
    if (i_cell == 0 || i_cell == Nx-1)
    {
      FLOAT x = i_cell == 0 ? X_min : X_max;
      IntC = d_C_Exact(x, v_d_y[ind], t);
    }
    else if (iR == iL) {
      IntC = d_fIntC(v_d_C6[id_R], v_d_Dltc[id_R], v_d_CL[id_R], xR_ksi) -
              d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], xL_ksi);
    } 
    else {    
      IntCL = d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], 1.0) -
              d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], xL_ksi);
      IntCR = d_fIntC(v_d_C6[id_R], v_d_Dltc[id_R], v_d_CL[id_R], xR_ksi);     
      IntCM = d_x_sum(v_d_C, iL + 1, iR - 1, j_cell, Ny);         
      IntC = IntCL + IntCM + IntCR;       
    }

    v_d_CRemap[ind] = IntC;

    ind += gridDim.x * blockDim.x;
  }
}

__global__ void y_Remap_on_gpu_part1(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6, FLOAT *v_d_Dltc, FLOAT *v_d_x, 
                                      FLOAT *v_d_y, FLOAT *v_d_C, INT N_grid, INT Nx, INT Ny, FLOAT t) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT bT, bB;
  int i_cell, j_cell;
  FLOAT Cn, Cnp1, C0m1, C0m2, Cj, Cjp1, Cjp2, Cjm1, Cjm2;

  FLOAT one_twelfth = 1.0 / 12.0;
  FLOAT siv_twelfth = 7.0 / 12.0;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    //************************ calculate ghost cell values ************************
    bB = d_C_Exact(v_d_x[ind], v_d_y[0], t);
    C0m1 = 1.0 / 3.0 * (-13 * v_d_C[i_cell * Ny] + 5 * v_d_C[i_cell * Ny + 1] - v_d_C[i_cell * Ny + 2] + 12 * bB);
    C0m2 = 1.0 / 3.0 * (-70 * v_d_C[i_cell * Ny] + 32 * v_d_C[i_cell * Ny + 1] - 7 * v_d_C[i_cell * Ny + 2] + 48 * bB);
    
    bT = d_C_Exact(v_d_x[ind], v_d_y[Ny-1], t);
    Cn = 1.0 / 3.0 * (-13 * v_d_C[i_cell * Ny + Ny-1] + 5 * v_d_C[i_cell * Ny + Ny-2] - v_d_C[i_cell * Ny + Ny-3] + 12 * bT);
    Cnp1 = 1.0 / 3.0 * (-70 * v_d_C[i_cell * Ny + Ny-1] + 32 * v_d_C[i_cell * Ny + Ny-2] - 7 * v_d_C[i_cell * Ny + Ny-3] + 48 * bT);

    //************************ set values for Cj, Cjp1, Cjp2, Cjm1, Cjm2 ************************
    Cj = v_d_C[i_cell * Ny + j_cell];
    Cjp1 = j_cell > Ny - 2 ? Cn : v_d_C[i_cell * Ny + j_cell + 1];
    Cjp2 = j_cell > Ny - 3 ? (j_cell == Ny - 2 ? Cn : Cnp1) : v_d_C[i_cell * Ny + j_cell + 2];
    Cjm1 = j_cell < 1 ? C0m1 : v_d_C[i_cell * Ny + j_cell - 1];
    Cjm2 = j_cell < 2 ? (j_cell == 1 ? C0m1 : C0m2) : v_d_C[i_cell * Ny + j_cell - 2];

    //************************ calculate CL, CR ************************
    v_d_CL[ind] = siv_twelfth * (Cj + Cjm1) - one_twelfth * (Cjp1 + Cjm2);
    v_d_CR[ind] = siv_twelfth * (Cjp1 + Cj) - one_twelfth * (Cjp2 + Cjm1);

    //**************** calculate_Dltc, C6 ***************
    v_d_Dltc[ind] = v_d_CR[ind] - v_d_CL[ind];
    v_d_C6[ind] = 6 * (v_d_C[ind] - (v_d_CL[ind] + v_d_CR[ind]) / 2);
    ind += gridDim.x * blockDim.x;
  } 
  return;
} 
   
__global__ void y_Remap_on_gpu_part2(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6,
                                     FLOAT *v_d_Dltc, FLOAT *v_d_C, FLOAT *v_d_CRemap,
                                     FLOAT *v_d_x, FLOAT *v_d_yhf, FLOAT *v_d_y_inter2, 
                                     INT N_grid, INT Nx, INT Ny, FLOAT dy, FLOAT t)
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0; // Initialize integral value to 0

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;

    FLOAT ybL = v_d_y_inter2[i_cell * (Ny+1) + j_cell];   // y_bar(i_cell-1/2);
    FLOAT ybR = v_d_y_inter2[i_cell * (Ny+1) + j_cell+1]; // y_bar(i_cell+1/2);

    FLOAT intybL = fmax(ybL, Y_min + 1e-16);
    FLOAT intybR = fmin(ybR, Y_max - 1e-16);
    INT jL = floor((intybL - Y_min) / dy);
    INT jR = floor((intybR - Y_min) / dy);
    FLOAT yL_ksi = (intybL - v_d_yhf[jL]) / dy;
    FLOAT yR_ksi = (intybR - v_d_yhf[jR]) / dy;

    INT id_R = i_cell * Ny + jR;
    INT id_L = i_cell * Ny + jL;
    //************************* Remap of C****************************
    
    if (j_cell == 0 || j_cell == Ny-1)
    {
      FLOAT y = j_cell == 0 ? Y_min : Y_max;
      IntC = d_C_Exact(v_d_x[ind], y, t); 
    }
    else if (jL == jR) {
      IntC = d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], yR_ksi) -
              d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], yL_ksi);
    } 
    else {
      IntCL = d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], 1) -
              d_fIntC(v_d_C6[id_L], v_d_Dltc[id_L], v_d_CL[id_L], yL_ksi);
      IntCR = d_fIntC(v_d_C6[id_R], v_d_Dltc[id_R], v_d_CL[id_R], yR_ksi);
      IntCM = d_y_sum(v_d_C, jL + 1, jR - 1, i_cell, Ny);
      IntC = IntCL + IntCM + IntCR;
    }  
       
    v_d_CRemap[ind] = IntC; 

    ind += gridDim.x * blockDim.x; 
  }
}

//*********************************
// Main Code
//*********************************

int main(int argc, char *argv[]) {
  // Start of the time controller
  // std::clock_t start;
  // start = std::clock();
  /*************************************
    Initialization of host variables
  *************************************/
  FLOAT T_min, T_max;
  FLOAT dx, dy, dt;
  FLOAT x, y, t0, t1;
  INT i_cell, j_cell, itime, tid;  
  INT Nx, Ny, Nt, T_span, N_grid, N_points, N_points_inter1, N_points_inter2;

  FLOAT E_2, E_Inf, Error, Cmassdif, cmass_initial, cmass_end;
  FLOAT *v_h_C0, *v_h_C1, *v_h_C_Exact;
  FLOAT *v_d_CL, *v_d_CR, *v_d_C6, *v_d_Dltc;
  FLOAT *v_d_C0, *v_d_C_xphf, *v_d_C0_Remap;              // step 1
  FLOAT *v_d_IntC_y0, *v_d_IntC_y0_Remap, *v_d_IntC_yp1; // step 2
  FLOAT *v_d_IntC_xphf;                                 // step 3
  FLOAT *v_d_x_inter1, *v_d_y_inter2, *v_d_x_inter3;
  FLOAT *v_d_x, *v_d_y, *v_d_xhf, *v_d_yhf, *v_d_xhf_bar1, *v_d_yhf_bar2, *v_d_xhf_bar3;
  
  if (argc < 2) {
    cout << "please input config file name" << endl;
  }

  /*************************************************
    read  parameters from config file
  *************************************************/
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

  // discretization
  dx = (X_max - X_min) / Nx;
  dy = (Y_max - Y_min) / Ny;
  dt = (T_max - T_min) / Nt;

  N_grid = Nx * Ny;
  N_points = (Nx + 1) * (Ny + 1);
  N_points_inter2 = Nx * (Ny + 1);
  N_points_inter1 = (Nx + 1) * Ny;

  FLOAT dt_half = 0.5 * dt;
  FLOAT dt_mul_dx_inverse = dt / dx;
  FLOAT dt_mul_dy_inverse = dt / dy;
  FLOAT dt_mul_dx_square_inverse = dt / (dx * dx);
  FLOAT dt_mul_dy_square_inverse = dt / (dy * dy);
  

  // GPU related variables and parameters
  int numBlocks;       // Number of blocks
  int threadsPerBlock; // Number of threads
  int maxThreadsPerBlock;
  cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	//maxThreadsPerBlock = prop.maxThreadsPerBlock;
  maxThreadsPerBlock = 128;
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

  // Allocate device memory for the location vector
  //*****************position vector******************
  cudaMalloc((void **)&v_d_x, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_y, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_xhf, N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_yhf, N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_xhf_bar1, N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_yhf_bar2, N_points * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_xhf_bar3, N_points * sizeof(FLOAT));

  //***************Intersection points***********************
  cudaMalloc((void **)&v_d_x_inter1, N_points_inter1 * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_y_inter2, N_points_inter2 * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_x_inter3, N_points_inter1 * sizeof(FLOAT));

  // Create device vector to store the solution at each time step
  //*********************** step 1 ***********************
  cudaMalloc((void **)&v_d_C0, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C_xphf, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C0_Remap, N_grid * sizeof(FLOAT));

  //*********************** step 2 ***********************
  cudaMalloc((void **)&v_d_IntC_y0, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_IntC_yp1, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_IntC_y0_Remap, N_grid * sizeof(FLOAT));

  //*********************** step 3 ***********************
  cudaMalloc((void **)&v_d_IntC_xphf, N_grid * sizeof(FLOAT));

  // Create device vector
  cudaMalloc((void **)&v_d_CL, N_grid* sizeof(FLOAT));
  cudaMalloc((void **)&v_d_CR, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C6, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Dltc, N_grid * sizeof(FLOAT));


  clock_t start = clock();
 // Calculte Initial condition and Exact solution on CPU
  clock_t t2 = clock();
  for (i_cell = 0; i_cell < Nx; i_cell++){
    x = X_min + (i_cell + 0.5) * dx;
      for (j_cell = 0; j_cell < Ny; j_cell++){
          y = Y_min + (j_cell + 0.5) * dy;
          tid = i_cell * Ny + j_cell;       
          v_h_C0[tid] = C_Exact(x, y, T_min);
          v_h_C_Exact[tid] = C_Exact(x, y, T_max);
      }
  } 
  cudaMemcpy(v_d_C0, v_h_C0, N_grid * sizeof(FLOAT), cudaMemcpyHostToDevice);
  output_result(v_h_C0, 0, Nx, Ny, dt);

  //***************************** parallel computation on GPU ****************************
  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  // Caluculate the location vector on GPU
  Get_Euler_points_on_gpu<<<numBlocks, threadsPerBlock>>>(v_d_x, v_d_y, Nx, Ny, N_grid, dx, dy);

  Get_Lagrange_points_on_gpu<<<numBlocks, threadsPerBlock>>>
  (v_d_xhf, v_d_yhf, v_d_xhf_bar1, v_d_yhf_bar2, v_d_xhf_bar3, Nx, Ny, N_points, dx, dy, dt); 
  cudaDeviceSynchronize();  

  //********************* step 1 **************************
  Intersection_point_vet_x<<<numBlocks, threadsPerBlock, 0, stream1>>>
  (v_d_xhf_bar1, v_d_yhf, v_d_y, v_d_x_inter1, Nx, Ny, N_points_inter1); 
  
  //********************* step 2 **************************
  Intersection_point_hori_y<<<numBlocks, threadsPerBlock, 0, stream2>>>
  (v_d_xhf, v_d_yhf_bar2, v_d_x, v_d_y_inter2, Nx, Ny, N_points_inter2); 
  
  //********************* step 3 **************************
  Intersection_point_vet_x<<<numBlocks, threadsPerBlock, 0, stream3>>>
  (v_d_xhf_bar3, v_d_yhf, v_d_y, v_d_x_inter3, Nx, Ny, N_points_inter1);   

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaStreamDestroy(stream3);
  
  for (itime = 1; itime <= Nt; itime++){
  //for (itime = 1; itime <= 1; itime++){
    t0 = (itime - 1) * dt;
    t1 = itime * dt;
    //****************************** step 1********************************
    //********** v_d_C0_Remap = Remap(v_d_C0) **********
    x_Remap_on_gpu_part1<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_x, v_d_y, v_d_C0, N_grid, Nx, Ny, t0);
    cudaDeviceSynchronize();
    x_Remap_on_gpu_part2<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_C0, v_d_C0_Remap, 
    v_d_xhf, v_d_y, v_d_x_inter1, N_grid, Nx, Ny, dx, t0);

    v_d_C_xphf = v_d_C0_Remap;
    cudaDeviceSynchronize();

    //******************************** step 2 ***********************************
    //************* v_d_Int_C = x_Remap(v_d_C) **********
    y_Remap_on_gpu_part1<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_x, v_d_y, v_d_C_xphf, N_grid, Nx, Ny, t1);
    y_Remap_on_gpu_part2<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_C_xphf, v_d_IntC_y0,
    v_d_x, v_d_yhf, v_d_y_inter2, N_grid, Nx, Ny, dy, t1);
    
    v_d_IntC_yp1 = v_d_IntC_y0;
    cudaDeviceSynchronize();

    //*********************************** step 3 **************************************
    //*************** caluculate CL, CR *****************
    x_Remap_on_gpu_part1<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_x, v_d_y, v_d_IntC_yp1, N_grid, Nx, Ny, t1);
    x_Remap_on_gpu_part2<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_IntC_yp1, v_d_IntC_xphf, 
    v_d_xhf, v_d_y, v_d_x_inter1, N_grid, Nx, Ny, dx, t1);
    
    //************************* Update the solution *******************************
    v_d_C0 = v_d_IntC_xphf;
    cudaDeviceSynchronize();
    
    // output result
    if (itime % T_span == 0) {
      cudaMemcpy(v_h_C1, v_d_C0, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
      output_result(v_h_C1, itime, Nx, Ny, dt);
    }
  
  }
  cudaMemcpy(v_h_C1, v_d_C0, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);


 // calculate error
  E_2 = 0.0;
  E_Inf = 0.0;
  Error = 0.0;
  Cmassdif = 0.0;
  cmass_end = 0.0;
  cmass_initial = 0.0;
  
  for (tid = 0; tid < N_grid; tid++) {
    Error = v_h_C1[tid] - v_h_C_Exact[tid];
    E_2 += Error * Error;
    E_Inf = fmax(E_Inf, fabs(Error));
    cmass_end = cmass_end + v_h_C1[tid];
    cmass_initial = cmass_initial + v_h_C0[tid];
  }
  cmass_end = cmass_end * dx * dy;
  cmass_initial = cmass_initial * dx * dy;
  Cmassdif = fabs(cmass_initial - cmass_end);
  E_2 = sqrt(E_2 * dx *dy);

  clock_t end = clock();
  
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
  //printf("sumf:%.15e\n", sumf * dx * dy);
  //printf("cmass_end:%.15e\n", cmass_end);
  //printf("cmass_initial:%.15e\n", cmass_initial);
  
  double time = (end - start) / (double)CLOCKS_PER_SEC;
  cout << "time:" << time << endl;


  // Free device memory
  cudaFree(v_d_x);
  cudaFree(v_d_y);
  cudaFree(v_d_xhf);
  cudaFree(v_d_yhf);
  cudaFree(v_d_xhf_bar1);
  cudaFree(v_d_yhf_bar2);
  cudaFree(v_d_xhf_bar3);
  cudaFree(v_d_x_inter1);
  cudaFree(v_d_y_inter2);
  cudaFree(v_d_x_inter3);

  cudaFree(v_d_C0);
  cudaFree(v_d_C_xphf);
  cudaFree(v_d_C0_Remap);
  cudaFree(v_d_IntC_y0);
  cudaFree(v_d_IntC_yp1);
  cudaFree(v_d_IntC_y0_Remap);
  cudaFree(v_d_IntC_xphf);

  cudaFree(v_d_CL);
  cudaFree(v_d_CR);
  cudaFree(v_d_C6);
  cudaFree(v_d_Dltc);

  // Free host memory
  free(v_h_C0);
  free(v_h_C1);
  free(v_h_C_Exact);

  return 0;
}