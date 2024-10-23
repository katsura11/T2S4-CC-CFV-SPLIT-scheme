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

__device__ FLOAT d_fIntC(FLOAT vC6, FLOAT vDltc, FLOAT vCL, FLOAT xi) {
  return -1.0 / 3 * vC6 * pow(xi, 3) + 1.0 / 2 * (vDltc + vC6) * pow(xi, 2) + vCL * xi;
}
__device__ FLOAT d_x_sum(FLOAT *v_C0, INT iL, INT iR, INT j, INT Ny) {
  FLOAT sum = 0.0;
  //INT i_start, i_end;
  //i_start = iL < iR ? iL : iR;
  //i_end = iL < iR ? iR : iL;
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

// output result to file
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
// calculate Exact solution
__global__ void get_exact_solution_on_gpu(FLOAT *v_d_x, FLOAT *v_d_y, FLOAT *v_d_C_Exact, 
                                          FLOAT t, FLOAT K, INT N_grid) 
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT x, y;
  while (tid < N_grid) {
    x = v_d_x[tid];
    y = v_d_y[tid];
    v_d_C_Exact[tid] = d_C_Exact(x, y, t, K);
    tid += gridDim.x * blockDim.x;
  }
}

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
    /***************************** step 1 *******************************/
    x_star1 = v_d_xhf[tid] - 0.5 * dt * d_Vx(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_xhf_bar1[tid] = v_d_xhf[tid] 
                        - 0.25 * dt * (d_Vx(v_d_xhf[tid], v_d_yhf[tid]) + d_Vx(x_star1, v_d_yhf[tid]));

    /***************************** step 2 *******************************/
    y_star2 = v_d_yhf[tid] -       dt * d_Vy(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_yhf_bar2[tid] = v_d_yhf[tid] 
                        -  0.5 * dt * (d_Vy(v_d_xhf[tid], v_d_yhf[tid]) + d_Vy(v_d_xhf[tid], y_star2));    
    
    /**************************** step 3 ********************************/
    x_star3 = v_d_xhf[tid] - 0.5 * dt * d_Vx(v_d_xhf[tid], v_d_yhf[tid]);
    v_d_xhf_bar3[tid] = v_d_xhf[tid] 
                        - 0.25 * dt * (d_Vx(v_d_xhf[tid], v_d_yhf[tid]) + d_Vx(x_star3, v_d_yhf[tid]));

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


// calculate Int(C) = Remap(C)
__global__ void x_Remap_on_gpu_part1(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6,
                                     FLOAT *v_d_Dltc, FLOAT *v_d_C, INT N_grid, INT Nx, INT Ny) 
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT one_twelfth = 1.0 / 12.0;
  FLOAT siv_twelfth = 7.0 / 12.0;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;
  
  while (ind < N_grid) {

    i_cell = ind / Ny;
    j_cell = ind % Ny;

    // ************************ calculate ghost cell values ************************
    Cn = one_elev * (-v_d_C[(Nx - 3) * Ny + j_cell] + 3 * v_d_C[(Nx - 2) * Ny + j_cell] 
                        + 9 * v_d_C[(Nx - 1) * Ny + j_cell]);
    Cnp1 = one_elev * (-15 * v_d_C[(Nx - 3) * Ny + j_cell] + 56 * v_d_C[(Nx - 2) * Ny + j_cell] 
                          - 30 * v_d_C[(Nx - 1) * Ny + j_cell]);
    C0m1 = one_elev * (-v_d_C[2 * Ny + j_cell] + 3 * v_d_C[Ny + j_cell] + 9 * v_d_C[j_cell]);
    C0m2 = one_elev * (-15 * v_d_C[2 * Ny + j_cell] + 56 * v_d_C[Ny + j_cell] - 30 * v_d_C[j_cell]);

    Ci = v_d_C[i_cell * Ny + j_cell]; // C(i,j)
    Cip1 = i_cell == Nx-1 ? Cn : v_d_C[(i_cell + 1) * Ny + j_cell]; // C(i+1,j)
    Cip2 = i_cell >= Nx-2 ? ((i_cell == Nx-2) ? Cn : Cnp1) : v_d_C[(i_cell + 2) * Ny + j_cell]; // C(i+2,j)
    Cim1 = i_cell == 0 ? C0m1 : v_d_C[(i_cell - 1) * Ny + j_cell]; // C(i-1,j)
    Cim2 = i_cell <= 1 ? ((i_cell == 1) ? C0m1 : C0m2) : v_d_C[(i_cell - 2) * Ny + j_cell]; // C(i-2,j)
    
    //************************ calculate CL, CR *************************** 
    v_d_CL[ind] = siv_twelfth * (Ci + Cim1) - one_twelfth * (Cip1 + Cim2); // C(i-1/2,j)
    v_d_CR[ind] = siv_twelfth * (Cip1 + Ci) - one_twelfth * (Cip2 + Cim1); // C(i+1/2,j)

    //**************** calculate Dltc, C6 ***************
    v_d_Dltc[ind] = v_d_CR[ind] - v_d_CL[ind]; 
    v_d_C6[ind] = 6.0 * (v_d_C[ind] - 0.5 * (v_d_CL[ind] + v_d_CR[ind]));
       
    ind += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void x_Remap_on_gpu_part2(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6,
                                     FLOAT *v_d_Dltc, FLOAT *v_d_C, FLOAT *v_d_CRemap,
                                     FLOAT *v_d_xhf, FLOAT *v_d_x_inter1, INT N_grid, 
                                     INT Nx, INT Ny, FLOAT dx) 
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0;

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;
    FLOAT xbL = v_d_x_inter1[i_cell * Ny + j_cell];
    FLOAT xbR = v_d_x_inter1[(i_cell + 1) * Ny + j_cell];

    // If xbL and xbR are outside the boundary or equal, then IntC = 0
    // Otherwise, calculate the integral value
    if (!(xbL > X_max || xbR < X_min || xbL == xbR)) {
      FLOAT intxbL = fmax(xbL, X_min + 1e-15);
      FLOAT intxbR = fmin(xbR, X_max - 1e-15);
      INT iL = floor((intxbL - X_min) / dx);
      INT iR = floor((intxbR - X_min) / dx);
      FLOAT xL_ksi = (intxbL - v_d_xhf[iL * (Ny + 1) + j_cell]) / dx;
      FLOAT xR_ksi = (intxbR - v_d_xhf[iR * (Ny + 1) + j_cell]) / dx;
      
      //************************* Remap of C****************************
      INT id_R = iR * Ny + j_cell;
      INT id_L = iL * Ny + j_cell;

      if (iR == iL) {
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
    }
    v_d_CRemap[ind] = IntC;

    ind += gridDim.x * blockDim.x;
  }
}

__global__ void y_Remap_on_gpu_part1(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6,
                                     FLOAT *v_d_Dltc, FLOAT *v_d_C, INT N_grid, INT Nx, INT Ny) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT one_twelfth = 1.0 / 12.0;
  FLOAT siv_twelfth = 7.0 / 12.0;
  FLOAT Cn, Cnp1, C0m1, C0m2, Cj, Cjp1, Cjp2, Cjm1, Cjm2;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    
    //************************ calculate ghost cell values ************************ 
    Cn = one_elev * (-v_d_C[i_cell * Ny + Ny-3] + 3 * v_d_C[i_cell * Ny + Ny-2] + 9 * v_d_C[i_cell * Ny + Ny-1]);
    Cnp1 = one_elev * (-15 * v_d_C[i_cell * Ny + Ny-3] + 56 * v_d_C[i_cell * Ny + Ny-2] - 30 * v_d_C[i_cell * Ny + Ny-1]);
    C0m1 = one_elev * (-v_d_C[i_cell * Ny + 2] + 3 * v_d_C[i_cell * Ny + 1] + 9 * v_d_C[i_cell * Ny]);
    C0m2 = one_elev * (-15 * v_d_C[i_cell * Ny + 2] + 56 * v_d_C[i_cell * Ny + 1] - 30 * v_d_C[i_cell * Ny]);
    
    Cj = v_d_C[i_cell * Ny + j_cell]; // C(i,j)
    Cjp1 = j_cell > Ny - 2 ? Cn : v_d_C[i_cell * Ny + j_cell + 1]; // C(i,j+1)
    Cjp2 = j_cell > Ny - 3 ? (j_cell == Ny - 2 ? Cn : Cnp1) : v_d_C[i_cell * Ny + j_cell + 2]; // C(i,j+2)
    Cjm1 = j_cell < 1 ? C0m1 : v_d_C[i_cell * Ny + j_cell - 1]; // C(i,j-1)
    Cjm2 = j_cell < 2 ? (j_cell == 1 ? C0m1 : C0m2) : v_d_C[i_cell * Ny + j_cell - 2]; // C(i,j-2)

    //************************ calculate CL, CR ************************
    v_d_CL[ind] = siv_twelfth * (Cj + Cjm1) - one_twelfth * (Cjp1 + Cjm2); // C(i,j-1/2)
    v_d_CR[ind] = siv_twelfth * (Cjp1 + Cj) - one_twelfth * (Cjp2 + Cjm1); // C(i,j+1/2)

    //**************** calculate_Dltc, C6 ***************
    v_d_Dltc[ind] = v_d_CR[ind] - v_d_CL[ind];
    v_d_C6[ind] = 6 * (v_d_C[ind] - (v_d_CL[ind] + v_d_CR[ind]) / 2);
    ind += gridDim.x * blockDim.x;
  } 
  return;
}
   
__global__ void y_Remap_on_gpu_part2(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C6,
                                     FLOAT *v_d_Dltc, FLOAT *v_d_C, FLOAT *v_d_CRemap,
                                     FLOAT *v_d_yhf, FLOAT *v_d_y_inter2, INT N_grid, 
                                     INT Nx, INT Ny, FLOAT dy)
{
  INT ind = blockDim.x * blockIdx.x + threadIdx.x;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0; // Initialize integral value to 0

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;

    FLOAT ybL = v_d_y_inter2[i_cell * (Ny+1) + j_cell];   // y_bar(i_cell-1/2);
    FLOAT ybR = v_d_y_inter2[i_cell * (Ny+1) + j_cell+1]; // y_bar(i_cell+1/2);

    // If ybL and ybR are outside the boundary or equal, then IntC = 0
    // Otherwise, calculate the integral value
    if (!(ybL > Y_max || ybR < Y_min || ybL == ybR)) {
      FLOAT intybL = fmax(ybL, Y_min + 1e-16);
      FLOAT intybR = fmin(ybR, Y_max - 1e-16);
      INT jL = floor((intybL - Y_min) / dy);
      INT jR = floor((intybR - Y_min) / dy);
      FLOAT yL_ksi = (intybL - v_d_yhf[jL]) / dy;
      FLOAT yR_ksi = (intybR - v_d_yhf[jR]) / dy;

      //************************* Remap of C****************************
      INT id_R = i_cell * Ny + jR;
      INT id_L = i_cell * Ny + jL;

      if (jL == jR) {
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
    }  
    v_d_CRemap[ind] = IntC; 

    ind += gridDim.x * blockDim.x; 
  }
}

// calculate Diff_term C(v_d_xhf_bar1)
__global__ void x_Diff_on_gpu(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C,
                              FLOAT *v_d_DC_x_bar, FLOAT *v_d_xhf,
                              FLOAT *v_d_x_inter1, FLOAT Kx, INT N_points_inter1, 
                              INT Nx, INT Ny, FLOAT dx) {
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT x_bar, xL_ksi, xL_ksi2, xL_ksi3;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;

  while (ind < N_points_inter1) {
    i_cell = ind / Ny;  // 0,1...,Nx
    j_cell = ind % Ny; // 0,1...,Ny-1

    //************************ calculate location **************************
    x_bar = v_d_x_inter1[i_cell * Ny + j_cell]; // x_bar(i_cell-1/2);
    
    //************************* calculate DC_x_bar *************************
    if (x_bar >= X_min && x_bar <= X_max)
    {
      FLOAT int_xb = fmin(fmax(x_bar, X_min + 1e-15), X_max - 1e-15);
      INT iL = floor((int_xb - X_min) / dx); // iL = 0, 1, ..., Nx-1
      xL_ksi = (int_xb - v_d_xhf[iL * (Ny + 1) + j_cell]) / dx;

      // Precompute xL_ksi powers
      xL_ksi2 = xL_ksi * xL_ksi;
      xL_ksi3 = xL_ksi2 * xL_ksi;

      // ghost cell values
      Cn = one_elev * (-v_d_C[(Nx - 3) * Ny + j_cell] + 3 * v_d_C[(Nx - 2) * Ny + j_cell] 
                          + 9 * v_d_C[(Nx - 1) * Ny + j_cell]);
      Cnp1 = one_elev * (-15 * v_d_C[(Nx - 3) * Ny + j_cell] + 56 * v_d_C[(Nx - 2) * Ny + j_cell] 
                            - 30 * v_d_C[(Nx - 1) * Ny + j_cell]);
      C0m1 = one_elev * (-v_d_C[2 * Ny + j_cell] + 3 * v_d_C[Ny + j_cell] + 9 * v_d_C[j_cell]);
      C0m2 = one_elev * (-15 * v_d_C[2 * Ny + j_cell] + 56 * v_d_C[Ny + j_cell] - 30 * v_d_C[j_cell]);

      Ci = v_d_C[iL * Ny + j_cell];
      Cip1 = iL > Nx - 2 ? Cn : v_d_C[(iL + 1) * Ny + j_cell];
      Cip2 = iL > Nx - 3 ? Cnp1 : (iL == Nx - 2 ? Cn : v_d_C[(iL + 2) * Ny + j_cell]);
      Cim1 = iL < 1 ? C0m1 : v_d_C[(iL - 1) * Ny + j_cell];
      Cim2 = iL < 2 ? C0m2 : (iL == 1 ? C0m1 : v_d_C[(iL - 2) * Ny + j_cell]);

      //***************************** calculate Diff term *****************************
      v_d_DC_x_bar[ind] = 1.0 / (12.0 * dx) *
                          ((2.0 * xL_ksi3 - 6.0 * xL_ksi2 + 3.0 * xL_ksi + 1.0) * Cim2 +
                          (-8.0 * xL_ksi3 + 18.0 * xL_ksi2 + 6.0 * xL_ksi - 15.0) * Cim1 +
                          (12.0 * xL_ksi3 - 18.0 * xL_ksi2 - 24.0 * xL_ksi + 15.0) * Ci +
                          (-8.0 * xL_ksi3 + 6.0 * xL_ksi2 + 18.0 * xL_ksi - 1.0) * Cip1 +
                          (2.0 * xL_ksi3 - 3.0 * xL_ksi) * Cip2);
    } 
    else {
      v_d_DC_x_bar[ind] = 0;
    }

    ind += gridDim.x * blockDim.x;
  }
  return;
}


__global__ void y_Diff_on_gpu(FLOAT *v_d_CL, FLOAT *v_d_CR, FLOAT *v_d_C, FLOAT *v_d_DC_y_bar, 
                              FLOAT *v_d_yhf, FLOAT *v_d_y_inter2, FLOAT Ky, INT N_points_inter2, 
                              INT Nx, INT Ny, FLOAT dy, FLOAT dt_mul_dy_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT yL_ksi, yL_ksi2, yL_ksi3;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;
  
  while (ind < N_points_inter2) {
    i_cell = ind / (Ny + 1);
    j_cell = ind % (Ny + 1);

    //***************************** calculate location *****************************
    FLOAT y_bar = v_d_y_inter2[i_cell * (Ny + 1) + j_cell];                                                                                

    //***************************** calculate DC_y_bar *****************************
    if (y_bar >= Y_min && y_bar <= Y_max) {
      FLOAT int_yb = fmin(fmax(y_bar, Y_min + 1e-15), Y_max - 1e-15);
      INT jL = floor((int_yb - Y_min) / dy);
      yL_ksi = (int_yb - v_d_yhf[i_cell * (Ny + 1) + jL]) / dy;

      //***************************** calculate Diff term *****************************
      // Precompute yL_ksi powers
      yL_ksi2 = yL_ksi * yL_ksi;
      yL_ksi3 = yL_ksi2 * yL_ksi;

      // ghost cell values
      Cn = one_elev * (-v_d_C[i_cell * Ny + Ny-3] + 3 * v_d_C[i_cell * Ny + Ny-2] + 9 * v_d_C[i_cell * Ny + Ny-1]);
      Cnp1 = one_elev * (-15 * v_d_C[i_cell * Ny + Ny-3] + 56 * v_d_C[i_cell * Ny + Ny-2] - 30 * v_d_C[i_cell * Ny + Ny-1]);
      C0m1 = one_elev * (-v_d_C[i_cell * Ny + 2] + 3 * v_d_C[i_cell * Ny + 1] + 9 * v_d_C[i_cell * Ny]);
      C0m2 = one_elev * (-15 * v_d_C[i_cell * Ny + 2] + 56 * v_d_C[i_cell * Ny + 1] - 30 * v_d_C[i_cell * Ny]);

      Ci = v_d_C[i_cell * Ny + jL];
      Cip1 = jL > Ny - 2 ? Cn : v_d_C[i_cell * Ny + jL + 1];
      Cip2 = jL > Ny - 3 ? Cnp1 : (jL == Ny - 2 ? Cn : v_d_C[i_cell * Ny + jL + 2]);
      Cim1 = jL < 1 ? C0m1 : v_d_C[i_cell * Ny + jL - 1];
      Cim2 = jL < 2 ? C0m2 : (jL == 1 ? C0m1 : v_d_C[i_cell * Ny + jL - 2]);

      v_d_DC_y_bar[ind] = 1.0 / (12.0 * dy) *
                          ((2.0 * yL_ksi3 - 6.0 * yL_ksi2 + 3.0 * yL_ksi + 1.0) * Cim2 +
                          (-8.0 * yL_ksi3 + 18.0 * yL_ksi2 + 6.0 * yL_ksi - 15.0) * Cim1 +
                          (12.0 * yL_ksi3 - 18.0 * yL_ksi2 - 24.0 * yL_ksi + 15.0) * Ci +
                          (-8.0 * yL_ksi3 + 6.0 * yL_ksi2 + 18.0 * yL_ksi - 1.0) * Cip1 +
                          (2.0 * yL_ksi3 - 3.0 * yL_ksi) * Cip2);
    }
    else {
      v_d_DC_y_bar[ind] = 0;
    }

    ind += gridDim.x * blockDim.x;
  }
  return;
}

// Allocate Matrix
__global__ void x_Create_matrix_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC, FLOAT *v_d_b, 
                                        FLOAT *v_d_CRemap, FLOAT *v_d_DC_x_bar, INT N_grid, INT Ny, 
                                        FLOAT Kx, FLOAT dt, FLOAT dt_mul_dx_inverse, 
                                        FLOAT dt_mul_dx_square_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    //********************** x direction ************************
    v_d_coA[ind] =  1.0/12 - 0.25 * Kx * dt_mul_dx_square_inverse;   //upper diagonal
    v_d_coB[ind] = 10.0/12 +  0.5 * Kx * dt_mul_dx_square_inverse;  //main diagonal
    v_d_coC[ind] =  1.0/12 - 0.25 * Kx * dt_mul_dx_square_inverse; //lower diagonal
    v_d_b[ind] = v_d_CRemap[ind]
                + 0.25 * Kx * dt_mul_dx_inverse * 
                (v_d_DC_x_bar[(i_cell + 1) * Ny + j_cell] - v_d_DC_x_bar[i_cell * Ny + j_cell]);
                 
    ind += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void y_Create_matrix_on_gpu(FLOAT *v_d_coA, FLOAT *v_d_coB, FLOAT *v_d_coC, FLOAT *v_d_b, 
                                        FLOAT *v_d_CRemap, FLOAT *v_d_DC_y_bar, INT N_grid, INT Ny, 
                                        FLOAT Ky, FLOAT dt, FLOAT dt_mul_dy_inverse, 
                                        FLOAT dt_mul_dy_square_inverse) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    //************************* y direction **************************
    v_d_coA[ind] =  1.0/12 - 0.5 * Ky * dt_mul_dy_square_inverse;   //upper diagonal
    v_d_coB[ind] = 10.0/12  + Ky * dt_mul_dy_square_inverse;       //main diagonal
    v_d_coC[ind] =  1.0/12 - 0.5 * Ky * dt_mul_dy_square_inverse; //lower diagonal
    v_d_b[ind] = v_d_CRemap[ind]
                + 0.5 * Ky * dt_mul_dy_inverse * 
                (v_d_DC_y_bar[i_cell * (Ny + 1) + j_cell+1] - v_d_DC_y_bar[i_cell * (Ny + 1) + j_cell]);

    ind += gridDim.x * blockDim.x;
  }
  return;
}


// calculate Compact Matrix
__global__ void x_Compact_operater_on_gpu(FLOAT *v_d_b, FLOAT *v_d_Cptb, INT N_grid, INT Ny, INT Nx) 
{
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell; 

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    if(i_cell == 0){
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell * Ny + j_cell] + v_d_b[(i_cell + 1) * Ny + j_cell]) / 12.0;
    }
    else if(i_cell == Nx-1)
    {
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell * Ny + j_cell] + v_d_b[(i_cell - 1) * Ny + j_cell]) / 12.0;
    }
    else{
        v_d_Cptb[ind] = (v_d_b[(i_cell + 1) * Ny + j_cell] + 10.0 * v_d_b[i_cell * Ny + j_cell] + v_d_b[(i_cell - 1) * Ny + j_cell]) / 12.0;
    }

    ind += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void y_Compact_operater_on_gpu(FLOAT *v_d_b, FLOAT *v_d_Cptb, INT N_grid, INT Ny) {
  int ind = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    if(j_cell == 0){
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell*Ny + j_cell] + v_d_b[i_cell*Ny + j_cell+1]) / 12.0;
    }
    else if(j_cell == Ny-1)
    {
        v_d_Cptb[ind] = (10.0 * v_d_b[i_cell*Ny + j_cell] + v_d_b[i_cell*Ny + j_cell-1]) / 12.0;
    }
    else{ 
        v_d_Cptb[ind] = (v_d_b[i_cell * Ny + (j_cell + 1)] + 10 * v_d_b[i_cell * Ny + j_cell] 
                        + v_d_b[i_cell * Ny + (j_cell - 1)]) / 12.0;
    }
    ind += gridDim.x * blockDim.x;
  }
  return;
}

// solve the equation using Thomas algorithm
__global__ void x_Thomas_on_gpu(FLOAT *v_d_A, FLOAT *v_d_B, FLOAT *v_d_C, FLOAT *v_d_b, 
                                FLOAT *v_d_C_xphf, FLOAT *p, FLOAT *q, INT N_grid, INT Nx, INT Ny) {

  INT ins = blockDim.x * blockIdx.x + threadIdx.x;
  INT i_cell, j_cell;
  FLOAT denom;
  while (ins < Ny) {
    j_cell = ins;

    // q[0]=q[0]/b[0] ; d[0]=d[0]/b[0];
    p[j_cell] = v_d_C[j_cell] / v_d_B[j_cell];
    q[j_cell] = (v_d_b[j_cell]) / v_d_B[j_cell];
    
    // Forward substitution
    for (i_cell = 1; i_cell < Nx; i_cell++) {
      denom = 1.0f / (v_d_B[i_cell * Ny + j_cell] - p[(i_cell - 1) * Ny + j_cell] * v_d_A[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_d_C[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_d_b[i_cell * Ny + j_cell] 
                                - q[(i_cell - 1) * Ny + j_cell] * v_d_A[i_cell * Ny + j_cell]) * denom;
    }

    // Backward substitution
    v_d_C_xphf[(Nx - 1) * Ny + j_cell] = q[(Nx - 1) * Ny + j_cell];
    for (int i_cell = Nx-2; i_cell >= 0; i_cell--) {
      v_d_C_xphf[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] 
                                        - p[i_cell * Ny + j_cell] * v_d_C_xphf[(i_cell + 1) * Ny + j_cell];
    }    
    ins += gridDim.x * blockDim.x;
  }
  return;
}

__global__ void y_Thomas_on_gpu(FLOAT *v_d_A, FLOAT *v_d_B, FLOAT *v_d_C, FLOAT *v_d_b, 
                                FLOAT *v_d_C_yp1, FLOAT *p, FLOAT *q, INT N_grid, INT Nx, INT Ny) 
{
  int ins = blockDim.x * blockIdx.x + threadIdx.x;
  int i_cell, j_cell;
  FLOAT denom;

  while (ins < Nx) {
    i_cell = ins ;

    // q[0]=q[0]/b[0] ; d[0]=d[0]/b[0];
    p[i_cell * Ny] = v_d_C[i_cell * Ny] / v_d_B[i_cell * Ny];
    q[i_cell * Ny] = (v_d_b[i_cell * Ny]) / v_d_B[i_cell * Ny];

    // Forward substitution
    for (int j_cell = 1; j_cell < Ny; j_cell++) {
      denom = 1.0f / (v_d_B[i_cell * Ny + j_cell] - p[i_cell * Ny + j_cell - 1] * v_d_A[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_d_C[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_d_b[i_cell * Ny + j_cell] 
                                - q[i_cell * Ny + j_cell - 1] * v_d_A[i_cell * Ny + j_cell]) * denom;
    }

    // Backward substitution
    v_d_C_yp1[i_cell * Ny + Ny-1] = q[i_cell * Ny + Ny-1];
    for (int j_cell = Ny-2; j_cell >= 0; j_cell--) {
      v_d_C_yp1[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] 
                                        - p[i_cell * Ny + j_cell] * v_d_C_yp1[i_cell * Ny + (j_cell + 1)];
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
  // std::clock_t start;
  // start = std::clock();
  /*************************************
    Initialization of host variables
  *************************************/
  INT T_span; // Time span of data output
  FLOAT T_min, T_max; // Time span
  FLOAT Kx, Ky; // Diffusion coefficients in x and y directions
  INT Nx, Ny, Nt; // Number of grid points in x and y directions and time steps
  FLOAT dx, dy, dt; // Grid spacing and time step size
  FLOAT x, y, t0, t1; // Variables for the location and time
  INT i_cell, j_cell, itime, tid;  // Variables for the cell index and time index
  INT N_grid, N_points, N_points_inter1, N_points_inter2; // Number of grid points and points for interpolation

  FLOAT E_2, E_Inf, Error, Cmassdif, cmass_initial, cmass_end; // Variables for the error and conservation of mass
  FLOAT *v_h_f, *v_h_C0, *v_h_C1, *v_h_C_Exact; // Variables for the host memory
  FLOAT *v_d_CL, *v_d_CR, *v_d_C6, *v_d_Dltc; // Variables for the cell values
  FLOAT *v_d_C0, *v_d_C_xphf, *v_d_C0_Remap, *v_d_DC0_x_bar; // Variables for the cell values in step 1
  FLOAT *v_d_IntC_y0, *v_d_IntC_y0_Remap, *v_d_DIntC_y0_bar, *v_d_IntC_yp1; // Variables for the cell values in step 2
  FLOAT *v_d_IntC_xphf, *v_d_DIntC_xphf_bar, *v_d_Cp1;  // Variables for the cell values in step 3
  FLOAT *v_d_x, *v_d_y, *v_d_xhf, *v_d_yhf, *v_d_xhf_bar1, *v_d_yhf_bar2, *v_d_xhf_bar3; // Variables for the location vector
  FLOAT *v_d_x_inter1, *v_d_y_inter2, *v_d_x_inter3; // Variables for the intersection points
  FLOAT *v_d_coA, *v_d_coB, *v_d_coC, *v_d_Cptb, *v_d_b, *p, *q; // Middle variables for the Thomas algorithm
  
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

  FLOAT dt_half = 0.5 * dt; // Half time step
  FLOAT dt_mul_dx_inverse = dt / dx; // dt/dx
  FLOAT dt_mul_dy_inverse = dt / dy; // dt/dy
  FLOAT dt_mul_dx_square_inverse = dt / (dx * dx); // dt/dx^2
  FLOAT dt_mul_dy_square_inverse = dt / (dy * dy); // dt/dy^2
  

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
  v_h_f = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
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
  cudaMalloc((void **)&v_d_DC0_x_bar, N_points_inter1 * sizeof(FLOAT));
  //*********************** step 2 ***********************
  cudaMalloc((void **)&v_d_IntC_y0, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_IntC_yp1, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_IntC_y0_Remap, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_DIntC_y0_bar, N_points_inter2 * sizeof(FLOAT));
  //*********************** step 3 ***********************
  cudaMalloc((void **)&v_d_Cp1, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_IntC_xphf, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_DIntC_xphf_bar, N_points_inter1 * sizeof(FLOAT));

  // Create device vector
  cudaMalloc((void **)&v_d_CL, N_grid* sizeof(FLOAT));
  cudaMalloc((void **)&v_d_CR, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_C6, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Dltc, N_grid * sizeof(FLOAT));

  // Create device vector to store matrix elements
  cudaMalloc((void **)&p, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&q, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_b, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_Cptb, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coA, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coB, N_grid * sizeof(FLOAT));
  cudaMalloc((void **)&v_d_coC, N_grid * sizeof(FLOAT));


  clock_t start = clock();
 // Calculte Initial condition and Exact solution on CPU
  clock_t t2 = clock();
  for (i_cell = 0; i_cell < Nx; i_cell++){
    x = X_min + (i_cell + 0.5) * dx;
      for (j_cell = 0; j_cell < Ny; j_cell++){
          y = Y_min + (j_cell + 0.5) * dy;
          tid = i_cell * Ny + j_cell;       
          v_h_C0[tid] = C_Exact(x, y, 0, Kx);
          v_h_C_Exact[tid] = C_Exact(x, y, T_max, Kx);
      }
  } 
  cudaMemcpy(v_d_C0, v_h_C0, N_grid * sizeof(FLOAT), cudaMemcpyHostToDevice);
  output_result(v_h_C0, 0, Nx, Ny, dt);

  //*********************** parallel computation on GPU ************************
  // Create streams
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

  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);
  cudaStreamSynchronize(stream3);

  cudaStreamDestroy(stream3);
  
  for (itime = 1; itime <= Nt; itime++){
  //for (itime = 1; itime <= 1; itime++){
    //cout << "itime = " << itime << endl;
    t0 = (itime - 1) * dt;
    t1 = itime * dt;
    //****************************** step 1********************************
    //********** v_d_C0_Remap = Remap(v_d_C0) **********
    x_Remap_on_gpu_part1<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_C0, N_grid, Nx, Ny);
    x_Remap_on_gpu_part2<<<numBlocks, threadsPerBlock, 0, stream1>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_C0, v_d_C0_Remap, 
    v_d_xhf, v_d_x_inter1, N_grid, Nx, Ny, dx);

    //*************** Diffusion term ********************
    x_Diff_on_gpu<<<numBlocks, threadsPerBlock, 0, stream2>>>
    (v_d_CL, v_d_CR, v_d_C0, v_d_DC0_x_bar, v_d_xhf, v_d_x_inter1, 
    Kx, N_points_inter1, Nx, Ny, dx);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    //*************** Matrix and vector******************
    x_Create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_C0_Remap, v_d_DC0_x_bar, 
    N_grid, Nx, Kx, dt, dt_mul_dx_inverse, dt_mul_dx_square_inverse);
    cudaDeviceSynchronize();
    
    //****************** Compact Scheme ******************
    x_Compact_operater_on_gpu<<<numBlocks, threadsPerBlock>>>(v_d_b, v_d_Cptb, N_grid, Ny, Nx);
    cudaDeviceSynchronize();
  
    //********* get the solution of x-direction **********
    x_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_Cptb, v_d_C_xphf, p, q, N_grid, Nx, Ny);
    cudaDeviceSynchronize();

    //******************************** step 2 ***********************************
    //************* v_d_Int_C = x_Remap(v_d_C) **********
    y_Remap_on_gpu_part1<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_C_xphf, N_grid, Nx, Ny);
    y_Remap_on_gpu_part2<<<numBlocks, threadsPerBlock, 0, stream1>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_C_xphf, v_d_IntC_y0,
    v_d_yhf, v_d_y_inter2, N_grid, Nx, Ny, dy);

    //**************** Diffusion term ********************
    y_Diff_on_gpu<<<numBlocks, threadsPerBlock, 0, stream2>>>
    (v_d_CL, v_d_CR, v_d_C_xphf, v_d_DIntC_y0_bar, v_d_yhf, v_d_y_inter2, 
    Ky, N_points_inter2, Nx, Ny, dy, dt_mul_dy_inverse);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    //*************** Matrix and vector******************
    y_Create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_IntC_y0, v_d_DIntC_y0_bar, 
    N_grid, Ny, Ky, dt, dt_mul_dy_inverse, dt_mul_dy_square_inverse);
    cudaDeviceSynchronize();
    
    //***************** Compact Scheme ******************
    y_Compact_operater_on_gpu<<<numBlocks, threadsPerBlock>>>(v_d_b, v_d_Cptb, N_grid, Ny);
    cudaDeviceSynchronize();
   
    //********* get the solution of y-direction *********
    y_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_Cptb, v_d_IntC_yp1, p, q, N_grid, Nx, Ny);
    cudaDeviceSynchronize();

    //*********************************** step 3 **************************************
    //*************** caluculate CL, CR *****************
    x_Remap_on_gpu_part1<<<numBlocks, threadsPerBlock>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_IntC_yp1, N_grid, Nx, Ny);
    x_Remap_on_gpu_part2<<<numBlocks, threadsPerBlock, 0, stream1>>>
    (v_d_CL, v_d_CR, v_d_C6, v_d_Dltc, v_d_IntC_yp1, v_d_IntC_xphf, 
    v_d_xhf, v_d_x_inter1, N_grid, Nx, Ny, dx);

    //****************** Diffusion term *****************
    x_Diff_on_gpu<<<numBlocks, threadsPerBlock, 0, stream2>>>
    (v_d_CL, v_d_CR, v_d_IntC_xphf, v_d_DIntC_xphf_bar, v_d_xhf, 
    v_d_x_inter3, Kx, N_points_inter1, Nx, Ny, dx);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    //*************** Matrix and vector******************
    x_Create_matrix_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_b, v_d_IntC_xphf, v_d_DIntC_xphf_bar, 
    N_grid, Nx, Kx, dt, dt_mul_dx_inverse, dt_mul_dx_square_inverse);
    cudaDeviceSynchronize();

    //**************** Compact Scheme ********************
    x_Compact_operater_on_gpu<<<numBlocks, threadsPerBlock>>>(v_d_b, v_d_Cptb, N_grid, Ny, Nx);
    cudaDeviceSynchronize();

    //********* get the solution of x-direction ***********
    x_Thomas_on_gpu<<<numBlocks, threadsPerBlock>>>
    (v_d_coA, v_d_coB, v_d_coC, v_d_Cptb, v_d_Cp1, p, q, N_grid, Nx, Ny);
    cudaDeviceSynchronize();
    
    //************************* Update the solution *******************************
    cudaMemcpy(v_d_C0, v_d_Cp1, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    
    // Output the result
    if (itime % T_span == 0) {
      cudaMemcpy(v_h_C1, v_d_C0, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
      output_result(v_h_C1, itime, Nx, Ny, dt);
    }
  
  }
  cudaMemcpy(v_h_C1, v_d_C0, N_grid * sizeof(FLOAT), cudaMemcpyDeviceToHost);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

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
  cudaFree(v_d_DC0_x_bar);
  cudaFree(v_d_IntC_y0);
  cudaFree(v_d_IntC_yp1);
  cudaFree(v_d_IntC_y0_Remap);
  cudaFree(v_d_DIntC_y0_bar);
  cudaFree(v_d_Cp1);
  cudaFree(v_d_IntC_xphf);
  cudaFree(v_d_DIntC_xphf_bar);

  cudaFree(v_d_CL);
  cudaFree(v_d_CR);
  cudaFree(v_d_C6);
  cudaFree(v_d_Dltc);
  cudaFree(v_d_coA);
  cudaFree(v_d_coB);
  cudaFree(v_d_coC);
  cudaFree(v_d_Cptb);
  cudaFree(v_d_b);
  cudaFree(p);
  cudaFree(q);

  // Free host memory
  free(v_h_f);
  free(v_h_C0);
  free(v_h_C1);
  free(v_h_C_Exact);


  return 0;
}