#include <iterator>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <fstream>

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
inline FLOAT Vx(FLOAT x, FLOAT y) { return -M_PI*cos(0.5*M_PI*x)*sin(0.5*M_PI*y); }
inline FLOAT Vy(FLOAT x, FLOAT y) { return  M_PI*cos(0.5*M_PI*y)*sin(0.5*M_PI*x); }
inline FLOAT C0(FLOAT x, FLOAT y) {
  return exp(-(pow((x - x0), 2) + pow((y - y0), 2)) / sigma);
}
inline FLOAT C_Exact(FLOAT x, FLOAT y, FLOAT t, FLOAT K) {
  FLOAT x_star =  x * cos(M_PI * t) + y * sin(M_PI * t);
  FLOAT y_star = -x * sin(M_PI * t) + y * cos(M_PI * t);
  return sigma / (sigma + 4 * K * t) *
         exp(-(pow((x_star - x0), 2) + pow((y_star - y0), 2)) / (sigma + 4 * K * t));
}
inline FLOAT f(FLOAT x, FLOAT y, FLOAT t, FLOAT K) {
    FLOAT x_star =  x * cos(M_PI * t) + y * sin(M_PI * t);
    FLOAT y_star = -x * sin(M_PI * t) + y * cos(M_PI * t);
    FLOAT p1 = sigma / (sigma + 4 * K * t) *
               exp(-(pow((x_star - x0), 2) + pow((y_star - y0), 2)) / (sigma + 4 * K * t));
    FLOAT p2 = (x - cos(0.5*M_PI*y) * sin(0.5*M_PI*x)) * ( x0*sin(M_PI*t) + y0*cos(M_PI*t) - y) 
             + (y - cos(0.5*M_PI*x) * sin(0.5*M_PI*y)) * (-x0*cos(M_PI*t) + y0*sin(M_PI*t) + x);

    return - M_PI * p1 * p2 /(sigma + 4 * K * t); 
}

inline FLOAT fIntC(FLOAT vC6, FLOAT vDltc, FLOAT vCL, FLOAT xi) {
  return -1.0 / 3 * vC6 * pow(xi, 3) + 1.0 / 2 * (vDltc + vC6) * pow(xi, 2) + vCL * xi;
}

inline FLOAT sum(FLOAT *v_d_C, INT N_grid) {
  FLOAT sum = 0.0;
  for (int idx = 0; idx <= N_grid; idx++) {
    sum += v_d_C[idx];
  }
  return sum;
}
inline FLOAT x_sum(FLOAT *v_C0, INT iL, INT iR, INT j, INT Ny) {
  FLOAT sum = 0.0;
  for (int i_cell = iL; i_cell <= iR; i_cell++) {
    sum += v_C0[i_cell * Ny + j];
  }
  return sum;
}
inline FLOAT y_sum(FLOAT *v_C0, INT jL, INT jR, INT i, INT Ny) {
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
void get_exact_solution(FLOAT *v_x, FLOAT *v_y, FLOAT *v_C_Exact, FLOAT t, FLOAT K, INT N_grid) 
{
  int tid = 0;
  FLOAT x, y;
  while (tid < N_grid) {
    x = v_x[tid];
    y = v_y[tid];
    v_C_Exact[tid] = C_Exact(x, y, t, K);
    tid ++;
  }
}

// calculate source term
void get_source_term(FLOAT *v_x, FLOAT *v_y, FLOAT *v_f, 
                                      FLOAT t, FLOAT K, INT N_grid) 
{
  int tid = 0;
  FLOAT x, y, x_star, y_star;
  FLOAT p1, p2, half_pi_x, half_pi_y;
  FLOAT a = sigma + 4 * K * t;
  FLOAT cos_PI_mul_t = cos(M_PI * t);
  FLOAT sin_PI_mul_t = sin(M_PI * t);

  while (tid < N_grid) {
    x = v_x[tid];
    y = v_y[tid];
    half_pi_x = 0.5 * M_PI * x;
    half_pi_y = 0.5 * M_PI * y;
    x_star =  x * cos_PI_mul_t + y * sin_PI_mul_t;
    y_star = -x * sin_PI_mul_t + y * cos_PI_mul_t;
    p1 = sigma / pow(a,2) * exp(-(pow((x_star - x0), 2) + pow((y_star - y0), 2)) / a);
    p2 = (x - cos(half_pi_y) * sin(half_pi_x)) * ( x0*sin_PI_mul_t + y0*cos_PI_mul_t - y) 
       + (y - cos(half_pi_x) * sin(half_pi_y)) * (-x0*cos_PI_mul_t + y0*sin_PI_mul_t + x);

    v_f[tid] = - M_PI * p1 * p2; 

    tid ++;
  }
}

// calculate int_points location
void Get_Euler_points(FLOAT *v_x, FLOAT *v_y, INT Nx, INT Ny, INT N_grid, FLOAT dx, FLOAT dy) 
{
  int tid = 0;
  INT i_cell, j_cell;

  while (tid < N_grid) {

    i_cell = tid / Ny; // i_cell = 0, 1, ..., Nx-1
    j_cell = tid % Ny; // i_cell = 0, 1, ..., Ny-1
    
    // calculate Eulerian points
    v_x[tid] = X_min + (i_cell + 0.5) * dx;
    v_y[tid] = Y_min + (j_cell + 0.5) * dy;

    tid ++;
  }
  return;
}

// calculate half_points location
void Get_Lagrange_points(FLOAT *v_xhf, FLOAT *v_yhf, FLOAT *v_xhf_bar1, 
                                          FLOAT *v_yhf_bar2, FLOAT *v_xhf_bar3, INT Nx, INT Ny, 
                                          INT N_points, FLOAT dx, FLOAT dy, FLOAT dt) 
{
  int tid = 0;
  FLOAT x_star1, y_star2, x_star3;
  INT i_point, j_point;

  while (tid < N_points) {

    i_point = tid / (Ny+1); // i_point = 0, 1, ..., Nx
    j_point = tid % (Ny+1); // i_point = 0, 1, ..., Ny

    // calculate Eulerian points
    v_xhf[tid] = X_min + dx * i_point;
    v_yhf[tid] = Y_min + dy * j_point;
    
    // calculate Lagrangian points
    /**************************************** step 1 *******************************************/
    x_star1 = v_xhf[tid] - 0.5 * dt * Vx(v_xhf[tid], v_yhf[tid]);
    v_xhf_bar1[tid] = v_xhf[tid] - 0.25 * dt * (Vx(v_xhf[tid], v_yhf[tid]) + Vx(x_star1, v_yhf[tid]));

    /**************************************** step 2 *******************************************/
    y_star2 = v_yhf[tid] -       dt * Vy(v_xhf[tid], v_yhf[tid]);
    v_yhf_bar2[tid] = v_yhf[tid] -  0.5 * dt * (Vy(v_xhf[tid], v_yhf[tid]) + Vy(v_xhf[tid], y_star2));    
    
    /**************************************** step 3 *******************************************/
    x_star3 = v_xhf[tid] - 0.5 * dt * Vx(v_xhf[tid], v_yhf[tid]);
    v_xhf_bar3[tid] = v_xhf[tid] - 0.25 * dt * (Vx(v_xhf[tid], v_yhf[tid]) + Vx(x_star3, v_yhf[tid]));

    tid ++;
  }
  return;
}


// Calculate intersection points in the x-direction
void Intersection_point_vet_x(FLOAT *v_xhf_bar1, FLOAT *v_yhf,
                                         FLOAT *v_y, FLOAT *v_x_inter1,
                                         INT Nx, INT Ny, INT N_points_inter1) 
{
  int tid = 0;

  while (tid < N_points_inter1) {
    INT j_cell = tid % Ny;   // j_cell = 0, 1, ..., Ny-1
    INT i_point = tid / Ny;  // i_point = 0, 1, ..., Nx
    INT ind = i_point * (Ny + 1) + j_cell;

    // Calculate intersection points
    FLOAT tmp = 1.0 / (v_yhf[ind] - v_yhf[ind + 1]);
    v_x_inter1[tid] = v_xhf_bar1[ind + 1] 
                        + (v_xhf_bar1[ind] -v_xhf_bar1[ind + 1]) * tmp
                        * (v_y[j_cell] - v_yhf[ind + 1]);

    tid ++;
  }
  return;
}

// Calculate intersection points in the y-direction
void Intersection_point_hori_y(FLOAT *v_xhf, FLOAT *v_yhf_bar2,
                                          FLOAT *v_x, FLOAT *v_y_inter2,
                                          INT Nx, INT Ny, INT N_points_inter2) 
{
  int tid = 0;

  while (tid < N_points_inter2) {
    INT i_cell = tid / (Ny + 1);   // i_cell = 0, 1, ..., Nx-1
    INT j_point = tid % (Ny + 1);  // j_point = 0, 1, ..., Ny
    
    INT ind = i_cell * (Ny + 1) + j_point;
    INT innext = (i_cell + 1) * (Ny + 1) + j_point;

    // Calculate intermediate x coordinate
    FLOAT x_int = 0.5 * (v_xhf[ind] + v_xhf[innext]);

    // Calculate intersection points
    FLOAT tmp = 1.0 / (v_xhf[ind] - v_xhf[innext]);
    v_y_inter2[tid] = v_yhf_bar2[innext] 
                      + (v_yhf_bar2[ind] - v_yhf_bar2[innext]) * tmp 
                      * (x_int - v_xhf[innext]);

    tid ++;
  }
  return;
}


// calculate Int(C) = Remap(C)
void x_Remap_part1(FLOAT *v_CL, FLOAT *v_CR, FLOAT *v_C6,
                                     FLOAT *v_Dltc, FLOAT *v_C, INT N_grid, INT Nx, INT Ny) 
{
  INT ind = 0;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT one_twelfth = 1.0 / 12.0;
  FLOAT siv_twelfth = 7.0 / 12.0;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;
  
  while (ind < N_grid) {

    i_cell = ind / Ny;
    j_cell = ind % Ny;

    // ************************ calculate ghost cell values ************************
    Cn = one_elev * (-v_C[(Nx - 3) * Ny + j_cell] + 3 * v_C[(Nx - 2) * Ny + j_cell] 
                      + 9 * v_C[(Nx - 1) * Ny + j_cell]);
    Cnp1 = one_elev * (-15 * v_C[(Nx - 3) * Ny + j_cell] + 56 * v_C[(Nx - 2) * Ny + j_cell] 
                        - 30 * v_C[(Nx - 1) * Ny + j_cell]);
    C0m1 = one_elev * (-v_C[2 * Ny + j_cell] + 3 * v_C[Ny + j_cell] + 9 * v_C[j_cell]);
    C0m2 = one_elev * (-15 * v_C[2 * Ny + j_cell] + 56 * v_C[Ny + j_cell] - 30 * v_C[j_cell]);

    Ci = v_C[i_cell * Ny + j_cell];
    Cip1 = i_cell == Nx-1 ? Cn : v_C[(i_cell + 1) * Ny + j_cell];
    Cip2 = i_cell >= Nx-2 ? ((i_cell == Nx-2) ? Cn : Cnp1) : v_C[(i_cell + 2) * Ny + j_cell];
    Cim1 = i_cell == 0 ? C0m1 : v_C[(i_cell - 1) * Ny + j_cell];
    Cim2 = i_cell <= 1 ? ((i_cell == 1) ? C0m1 : C0m2) : v_C[(i_cell - 2) * Ny + j_cell];
    
    //************************ calculate CL, CR *************************** 
    v_CL[ind] = siv_twelfth * (Ci + Cim1) - one_twelfth * (Cip1 + Cim2);
    v_CR[ind] = siv_twelfth * (Cip1 + Ci) - one_twelfth * (Cip2 + Cim1);

    //**************** calculate Dltc, C6 ***************
    v_Dltc[ind] = v_CR[ind] - v_CL[ind];
    v_C6[ind] = 6.0 * (v_C[ind] - 0.5 * (v_CL[ind] + v_CR[ind]));
       
    ind ++;
  }
  return;
}

void x_Remap_part2(FLOAT *v_CL, FLOAT *v_CR, FLOAT *v_C6,
                                     FLOAT *v_Dltc, FLOAT *v_C, FLOAT *v_CRemap,
                                     FLOAT *v_xhf, FLOAT *v_x_inter1, INT N_grid, 
                                     INT Nx, INT Ny, FLOAT dx) 
{
  INT ind = 0;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0;

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;
    FLOAT xbL = v_x_inter1[i_cell * Ny + j_cell];
    FLOAT xbR = v_x_inter1[(i_cell + 1) * Ny + j_cell];

    // If xbL and xbR are outside the boundary or equal, then IntC = 0
    // Otherwise, calculate the integral value
    if (!(xbL > X_max || xbR < X_min || xbL == xbR)) {
      FLOAT intxbL = fmax(xbL, X_min + 1e-15);
      FLOAT intxbR = fmin(xbR, X_max - 1e-15);
      INT il = floor((intxbL - X_min) / dx);
      INT ir = floor((intxbR - X_min) / dx);
      FLOAT xL_ksi = (intxbL - v_xhf[il * (Ny + 1) + j_cell]) / dx;
      FLOAT xR_ksi = (intxbR - v_xhf[ir * (Ny + 1) + j_cell]) / dx;
      
      //************************* Remap of C****************************
      INT iR = ir * Ny + j_cell;
      INT iL = il * Ny + j_cell;

      if (iR == iL) {
        IntC = fIntC(v_C6[iR], v_Dltc[iR], v_CL[iR], xR_ksi) -
                fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], xL_ksi);
      } 
      else {    
        IntCL = fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], 1.0) -
                fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], xL_ksi);
        IntCR = fIntC(v_C6[iR], v_Dltc[iR], v_CL[iR], xR_ksi);     
        IntCM = x_sum(v_C, iL + 1, iR - 1, j_cell, Ny);         
        IntC = IntCL + IntCM + IntCR;       
      }
    }
    v_CRemap[ind] = IntC;

    ind ++;
  }
}

void y_Remap_part1(FLOAT *v_CL, FLOAT *v_CR, FLOAT *v_C6,
                                     FLOAT *v_Dltc, FLOAT *v_C, INT N_grid, INT Nx, INT Ny) 
{
  INT ind = 0;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT one_twelfth = 1.0 / 12.0;
  FLOAT siv_twelfth = 7.0 / 12.0;
  FLOAT Cn, Cnp1, C0m1, C0m2, Cj, Cjp1, Cjp2, Cjm1, Cjm2;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    
    //************************ calculate ghost cell values ************************ 
    Cn = one_elev * (-v_C[i_cell * Ny + Ny-3] + 3 * v_C[i_cell * Ny + Ny-2] + 9 * v_C[i_cell * Ny + Ny-1]);
    Cnp1 = one_elev * (-15 * v_C[i_cell * Ny + Ny-3] + 56 * v_C[i_cell * Ny + Ny-2] - 30 * v_C[i_cell * Ny + Ny-1]);
    C0m1 = one_elev * (-v_C[i_cell * Ny + 2] + 3 * v_C[i_cell * Ny + 1] + 9 * v_C[i_cell * Ny]);
    C0m2 = one_elev * (-15 * v_C[i_cell * Ny + 2] + 56 * v_C[i_cell * Ny + 1] - 30 * v_C[i_cell * Ny]);

    Cj = v_C[i_cell * Ny + j_cell];
    Cjp1 = j_cell > Ny - 2 ? Cn : v_C[i_cell * Ny + j_cell + 1];
    Cjp2 = j_cell > Ny - 3 ? (j_cell == Ny - 2 ? Cn : Cnp1) : v_C[i_cell * Ny + j_cell + 2];
    Cjm1 = j_cell < 1 ? C0m1 : v_C[i_cell * Ny + j_cell - 1];
    Cjm2 = j_cell < 2 ? (j_cell == 1 ? C0m1 : C0m2) : v_C[i_cell * Ny + j_cell - 2];

    //************************ calculate CL, CR ************************
    v_CL[ind] = siv_twelfth * (Cj + Cjm1) - one_twelfth * (Cjp1 + Cjm2);
    v_CR[ind] = siv_twelfth * (Cjp1 + Cj) - one_twelfth * (Cjp2 + Cjm1);

    //**************** calculate_Dltc, C6 ***************
    v_Dltc[ind] = v_CR[ind] - v_CL[ind];
    v_C6[ind] = 6 * (v_C[ind] - (v_CL[ind] + v_CR[ind]) / 2);
    ind ++;
  } 
  return;
} 
   
void y_Remap_part2(FLOAT *v_CL, FLOAT *v_CR, FLOAT *v_C6,
                                     FLOAT *v_Dltc, FLOAT *v_C, FLOAT *v_CRemap,
                                     FLOAT *v_yhf, FLOAT *v_y_inter2, INT N_grid, 
                                     INT Nx, INT Ny, FLOAT dy)
{
  INT ind = 0;
  FLOAT IntCL, IntCR, IntCM;
  FLOAT IntC = 0.0; // Initialize integral value to 0

  while (ind < N_grid) {
    INT i_cell = ind / Ny;
    INT j_cell = ind % Ny;

    FLOAT ybL = v_y_inter2[i_cell * (Ny+1) + j_cell];   // y_bar(i_cell-1/2);
    FLOAT ybR = v_y_inter2[i_cell * (Ny+1) + j_cell+1]; // y_bar(i_cell+1/2);

    // If ybL and ybR are outside the boundary or equal, then IntC = 0
    // Otherwise, calculate the integral value
    if (!(ybL > Y_max || ybR < Y_min || ybL == ybR)) {
      FLOAT intybL = fmax(ybL, Y_min + 1e-16);
      FLOAT intybR = fmin(ybR, Y_max - 1e-16);
      INT jL = floor((intybL - Y_min) / dy);
      INT jR = floor((intybR - Y_min) / dy);
      FLOAT yL_ksi = (intybL - v_yhf[jL]) / dy;
      FLOAT yR_ksi = (intybR - v_yhf[jR]) / dy;

      //************************* Remap of C****************************
      INT iR = i_cell * Ny + jR;
      INT iL = i_cell * Ny + jL;

      if (jL == jR) {
        IntC = fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], yR_ksi) -
               fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], yL_ksi);
      } 
      else {
        IntCL = fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], 1) -
                fIntC(v_C6[iL], v_Dltc[iL], v_CL[iL], yL_ksi);
        IntCR = fIntC(v_C6[iR], v_Dltc[iR], v_CL[iR], yR_ksi);
        IntCM = y_sum(v_C, jL + 1, jR - 1, i_cell, Ny);
        IntC = IntCL + IntCM + IntCR;
      }  
    }  
    v_CRemap[ind] = IntC; 

    ind ++; 
  }
}

// calculate Diff_term C(v_xhf_bar1)
void x_Diff(FLOAT *v_CL, FLOAT *v_CR, FLOAT *v_C,
                              FLOAT *v_DC_x_bar, FLOAT *v_xhf,
                              FLOAT *v_x_inter1, FLOAT Kx, INT N_points_inter1, 
                              INT Nx, INT Ny, FLOAT dx) {
  INT ind = 0;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT x_bar, xL_ksi, xL_ksi2, xL_ksi3;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;

  while (ind < N_points_inter1) {
    i_cell = ind / Ny;  // 0,1...,Nx
    j_cell = ind % Ny; // 0,1...,Ny-1

    //************************ calculate location **************************
    x_bar = v_x_inter1[i_cell * Ny + j_cell]; // x_bar(i_cell-1/2);
    
    //************************* calculate DC_x_bar *************************
    if (x_bar >= X_min && x_bar <= X_max)
    {
      FLOAT int_xb = fmin(fmax(x_bar, X_min + 1e-15), X_max - 1e-15);
      INT iL = floor((int_xb - X_min) / dx); // iL = 0, 1, ..., Nx-1
      xL_ksi = (int_xb - v_xhf[iL * (Ny + 1) + j_cell]) / dx;

      // Precompute xL_ksi powers
      xL_ksi2 = xL_ksi * xL_ksi;
      xL_ksi3 = xL_ksi2 * xL_ksi;

      // ghost cell values
      Cn = one_elev * (-v_C[(Nx - 3) * Ny + j_cell] + 3 * v_C[(Nx - 2) * Ny + j_cell] 
                          + 9 * v_C[(Nx - 1) * Ny + j_cell]);
      Cnp1 = one_elev * (-15 * v_C[(Nx - 3) * Ny + j_cell] + 56 * v_C[(Nx - 2) * Ny + j_cell] 
                            - 30 * v_C[(Nx - 1) * Ny + j_cell]);
      C0m1 = one_elev * (-v_C[2 * Ny + j_cell] + 3 * v_C[Ny + j_cell] + 9 * v_C[j_cell]);
      C0m2 = one_elev * (-15 * v_C[2 * Ny + j_cell] + 56 * v_C[Ny + j_cell] - 30 * v_C[j_cell]);

      Ci = v_C[iL * Ny + j_cell];
      Cip1 = iL > Nx - 2 ? Cn : v_C[(iL + 1) * Ny + j_cell];
      Cip2 = iL > Nx - 3 ? Cnp1 : (iL == Nx - 2 ? Cn : v_C[(iL + 2) * Ny + j_cell]);
      Cim1 = iL < 1 ? C0m1 : v_C[(iL - 1) * Ny + j_cell];
      Cim2 = iL < 2 ? C0m2 : (iL == 1 ? C0m1 : v_C[(iL - 2) * Ny + j_cell]);

      //***************************** calculate Diff term *****************************
      v_DC_x_bar[ind] = 1.0 / (12.0 * dx) *
                          ((2.0 * xL_ksi3 - 6.0 * xL_ksi2 + 3.0 * xL_ksi + 1.0) * Cim2 +
                          (-8.0 * xL_ksi3 + 18.0 * xL_ksi2 + 6.0 * xL_ksi - 15.0) * Cim1 +
                          (12.0 * xL_ksi3 - 18.0 * xL_ksi2 - 24.0 * xL_ksi + 15.0) * Ci +
                          (-8.0 * xL_ksi3 + 6.0 * xL_ksi2 + 18.0 * xL_ksi - 1.0) * Cip1 +
                          (2.0 * xL_ksi3 - 3.0 * xL_ksi) * Cip2);
    } 
    else {
      v_DC_x_bar[ind] = 0;
    }

    ind ++;
  }
  return;
}


void y_Diff(FLOAT *v_CL, FLOAT *v_CR, FLOAT *v_C, FLOAT *v_DC_y_bar, 
                              FLOAT *v_yhf, FLOAT *v_y_inter2, FLOAT Ky, INT N_points_inter2, 
                              INT Nx, INT Ny, FLOAT dy, FLOAT dt_mul_dy_inverse) 
{
  INT ind = 0;
  int i_cell, j_cell;
  FLOAT one_elev = 1.0 / 11.0;
  FLOAT yL_ksi, yL_ksi2, yL_ksi3;
  FLOAT Cn, Cnp1, C0m1, C0m2, Ci, Cip1, Cip2, Cim1, Cim2;
  
  while (ind < N_points_inter2) {
    i_cell = ind / (Ny + 1);
    j_cell = ind % (Ny + 1);

    //***************************** calculate location *****************************
    FLOAT y_bar = v_y_inter2[i_cell * (Ny + 1) + j_cell];                                                                                

    //***************************** calculate DC_y_bar *****************************
    if (y_bar >= Y_min && y_bar <= Y_max) {
      FLOAT int_yb = fmin(fmax(y_bar, Y_min + 1e-15), Y_max - 1e-15);
      INT jL = floor((int_yb - Y_min) / dy);
      yL_ksi = (int_yb - v_yhf[i_cell * (Ny + 1) + jL]) / dy;

      //***************************** calculate Diff term *****************************
      // Precompute yL_ksi powers
      yL_ksi2 = yL_ksi * yL_ksi;
      yL_ksi3 = yL_ksi2 * yL_ksi;

      // Boundary interpolation values
      Cn = one_elev * (-v_C[i_cell * Ny + Ny-3] + 3 * v_C[i_cell * Ny + Ny-2] + 9 * v_C[i_cell * Ny + Ny-1]);
      Cnp1 = one_elev * (-15 * v_C[i_cell * Ny + Ny-3] + 56 * v_C[i_cell * Ny + Ny-2] - 30 * v_C[i_cell * Ny + Ny-1]);
      C0m1 = one_elev * (-v_C[i_cell * Ny + 2] + 3 * v_C[i_cell * Ny + 1] + 9 * v_C[i_cell * Ny]);
      C0m2 = one_elev * (-15 * v_C[i_cell * Ny + 2] + 56 * v_C[i_cell * Ny + 1] - 30 * v_C[i_cell * Ny]);

      Ci = v_C[i_cell * Ny + jL];
      Cip1 = jL > Ny - 2 ? Cn : v_C[i_cell * Ny + jL + 1];
      Cip2 = jL > Ny - 3 ? Cnp1 : (jL == Ny - 2 ? Cn : v_C[i_cell * Ny + jL + 2]);
      Cim1 = jL < 1 ? C0m1 : v_C[i_cell * Ny + jL - 1];
      Cim2 = jL < 2 ? C0m2 : (jL == 1 ? C0m1 : v_C[i_cell * Ny + jL - 2]);

      v_DC_y_bar[ind] = 1.0 / (12.0 * dy) *
                          ((2.0 * yL_ksi3 - 6.0 * yL_ksi2 + 3.0 * yL_ksi + 1.0) * Cim2 +
                          (-8.0 * yL_ksi3 + 18.0 * yL_ksi2 + 6.0 * yL_ksi - 15.0) * Cim1 +
                          (12.0 * yL_ksi3 - 18.0 * yL_ksi2 - 24.0 * yL_ksi + 15.0) * Ci +
                          (-8.0 * yL_ksi3 + 6.0 * yL_ksi2 + 18.0 * yL_ksi - 1.0) * Cip1 +
                          (2.0 * yL_ksi3 - 3.0 * yL_ksi) * Cip2);
    }
    else {
      v_DC_y_bar[ind] = 0;
    }

    ind ++;
  }
  return;
}

// Allocate Matrix
void x_Create_matrix(FLOAT *v_coA, FLOAT *v_coB, FLOAT *v_coC, FLOAT *v_b, 
                                        FLOAT *v_CRemap, FLOAT *v_DC_x_bar, FLOAT *v_f_p0, 
                                        FLOAT *v_f_p1, INT N_grid, INT Ny, FLOAT Kx, FLOAT dt, 
                                        FLOAT dt_mul_dx_inverse, FLOAT dt_mul_dx_square_inverse) 
{
  INT ind = 0;
  int i_cell, j_cell;
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    //********************** x direction ************************
    v_coA[ind] =  1.0/12 - 0.25 * Kx * dt_mul_dx_square_inverse;   //upper diagonal
    v_coB[ind] = 10.0/12 +  0.5 * Kx * dt_mul_dx_square_inverse;  //main diagonal
    v_coC[ind] =  1.0/12 - 0.25 * Kx * dt_mul_dx_square_inverse; //lower diagonal
    v_b[ind] = v_CRemap[ind] + 0.25 * dt * (v_f_p0[ind] + v_f_p1[ind])
                + 0.25 * Kx * dt_mul_dx_inverse * 
                (v_DC_x_bar[(i_cell + 1) * Ny + j_cell] - v_DC_x_bar[i_cell * Ny + j_cell]);
                 
    ind ++;
  }
  return;
}

void y_Create_matrix(FLOAT *v_coA, FLOAT *v_coB, FLOAT *v_coC, FLOAT *v_b, 
                                        FLOAT *v_CRemap, FLOAT *v_DC_y_bar, FLOAT *v_f_p0, 
                                        FLOAT *v_f_p1, INT N_grid, INT Ny, FLOAT Ky, FLOAT dt, 
                                        FLOAT dt_mul_dy_inverse, FLOAT dt_mul_dy_square_inverse) 
{
  INT ind = 0;
  int i_cell, j_cell;
  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;

    //************************* y direction **************************
    v_coA[ind] =  1.0/12 - 0.5 * Ky * dt_mul_dy_square_inverse;   //upper diagonal
    v_coB[ind] = 10.0/12  + Ky * dt_mul_dy_square_inverse;       //main diagonal
    v_coC[ind] =  1.0/12 - 0.5 * Ky * dt_mul_dy_square_inverse; //lower diagonal
    v_b[ind] = v_CRemap[ind] + 0.5 * dt * (v_f_p0[ind] + v_f_p1[ind])
                + 0.5 * Ky * dt_mul_dy_inverse * 
                (v_DC_y_bar[i_cell * (Ny + 1) + j_cell+1] - v_DC_y_bar[i_cell * (Ny + 1) + j_cell]);

    ind ++;
  }
  return;
}


// calculate Compact Matrix
void x_Compact_operater(FLOAT *v_b, FLOAT *v_Cptb, INT N_grid, INT Ny, INT Nx) 
{
  INT ind = 0;
  int i_cell, j_cell; 

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    if(i_cell == 0){
        v_Cptb[ind] = (10.0 * v_b[i_cell * Ny + j_cell] + 2 * v_b[(i_cell + 1) * Ny + j_cell]) / 12.0;
    }
    else if(i_cell == Nx-1)
    {
        v_Cptb[ind] = (10.0 * v_b[i_cell * Ny + j_cell] + 2 * v_b[(i_cell - 1) * Ny + j_cell]) / 12.0;
    }
    else{
        v_Cptb[ind] = (v_b[(i_cell + 1) * Ny + j_cell] + 10.0 * v_b[i_cell * Ny + j_cell] + v_b[(i_cell - 1) * Ny + j_cell]) / 12.0;
    }

    ind ++;
  }
  return;
}

void y_Compact_operater(FLOAT *v_b, FLOAT *v_Cptb, INT N_grid, INT Ny) {
  INT ind = 0;
  int i_cell, j_cell;

  while (ind < N_grid) {
    i_cell = ind / Ny;
    j_cell = ind % Ny;
    if(j_cell == 0){
        v_Cptb[ind] = (10 * v_b[i_cell*Ny + j_cell] + 2 * v_b[i_cell*Ny + j_cell+1]) / 12.0;
    }
    else if(j_cell == Ny-1)
    {
        v_Cptb[ind] = (10 * v_b[i_cell*Ny + j_cell] + 2 * v_b[i_cell*Ny + j_cell-1]) / 12.0;
    }
    else{ 
        v_Cptb[ind] = (v_b[i_cell * Ny + (j_cell + 1)] + 10 * v_b[i_cell * Ny + j_cell] + v_b[i_cell * Ny + (j_cell - 1)]) / 12.0;
    }
    ind ++;
  }
  return;
}

// solve the equation using Thomas algorithm
void x_Thomas(FLOAT *v_A, FLOAT *v_B, FLOAT *v_C, FLOAT *v_b, 
                                FLOAT *v_C_xphf, FLOAT *p, FLOAT *q, INT N_grid, INT Nx, INT Ny) {

  INT ins = 0;
  INT i_cell, j_cell;
  FLOAT denom;
  while (ins < Ny) {
    j_cell = ins;

    // q[0]=q[0]/b[0] ; d[0]=d[0]/b[0];
    p[j_cell] = v_C[j_cell] / v_B[j_cell];
    q[j_cell] = (v_b[j_cell]) / v_B[j_cell];
    
    // Forward substitution
    for (i_cell = 1; i_cell < Nx; i_cell++) {
      denom = 1.0f / (v_B[i_cell * Ny + j_cell] - p[(i_cell - 1) * Ny + j_cell] * v_A[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_C[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_b[i_cell * Ny + j_cell] 
                                - q[(i_cell - 1) * Ny + j_cell] * v_A[i_cell * Ny + j_cell]) * denom;
    }

    // Backward substitution
    v_C_xphf[(Nx - 1) * Ny + j_cell] = q[(Nx - 1) * Ny + j_cell];
    for (int i_cell = Nx-2; i_cell >= 0; i_cell--) {
      v_C_xphf[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] 
                                        - p[i_cell * Ny + j_cell] * v_C_xphf[(i_cell + 1) * Ny + j_cell];
    }    
    ins ++;
  }
  return;
}

void y_Thomas(FLOAT *v_A, FLOAT *v_B, FLOAT *v_C, FLOAT *v_b, 
                                FLOAT *v_C_yp1, FLOAT *p, FLOAT *q, INT N_grid, INT Nx, INT Ny) 
{
  INT ins = 0;
  int i_cell, j_cell;
  FLOAT denom;

  while (ins < Nx) {
    i_cell = ins ;

    // q[0]=q[0]/b[0] ; d[0]=d[0]/b[0];
    p[i_cell * Ny] = v_C[i_cell * Ny] / v_B[i_cell * Ny];
    q[i_cell * Ny] = (v_b[i_cell * Ny]) / v_B[i_cell * Ny];

    // Forward substitution
    for (int j_cell = 1; j_cell < Ny; j_cell++) {
      denom = 1.0f / (v_B[i_cell * Ny + j_cell] - p[i_cell * Ny + j_cell - 1] * v_A[i_cell * Ny + j_cell]);
      p[i_cell * Ny + j_cell] = v_C[i_cell * Ny + j_cell] * denom;
      q[i_cell * Ny + j_cell] = (v_b[i_cell * Ny + j_cell] 
                                - q[i_cell * Ny + j_cell - 1] * v_A[i_cell * Ny + j_cell]) * denom;
    }

    // Backward substitution
    v_C_yp1[i_cell * Ny + Ny-1] = q[i_cell * Ny + Ny-1];
    for (int j_cell = Ny-2; j_cell >= 0; j_cell--) {
      v_C_yp1[i_cell * Ny + j_cell] = q[i_cell * Ny + j_cell] 
                                        - p[i_cell * Ny + j_cell] * v_C_yp1[i_cell * Ny + (j_cell + 1)];
    }  
    ins ++;
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
  FLOAT Kx, Ky;
  FLOAT dx, dy, dt;
  FLOAT x, y, t0, t1;
  INT i_cell, j_cell, itime, tid; 
  FLOAT T_min, T_max; 
  INT Nx, Ny, Nt, T_span, N_grid, N_points, N_points_inter1, N_points_inter2;

  FLOAT E_2, E_Inf, Error, Cmassdif, cmass_initial, cmass_end;
  FLOAT *v_h_f, *v_h_C1, *v_h_C_Exact;
  FLOAT *v_CL, *v_CR, *v_C6, *v_Dltc;
  FLOAT *v_C0, *v_C_xphf, *v_C0_Remap, *v_DC0_x_bar;                 // step 1
  FLOAT *v_IntC_y0, *v_IntC_y0_Remap, *v_DIntC_y0_bar, *v_IntC_yp1; // step 2
  FLOAT *v_IntC_xphf, *v_DIntC_xphf_bar, *v_Cp1;                     // step 3
  FLOAT *v_x, *v_y, *v_xhf, *v_yhf, *v_xhf_bar1, *v_yhf_bar2, *v_xhf_bar3;
  FLOAT *v_x_inter1, *v_y_inter2, *v_x_inter3;
  FLOAT *v_coA, *v_coB, *v_coC, *v_Cptb, *v_b, *p, *q;
  FLOAT *v_f1_p0, *v_f1_p1, *v_Int_f1_p0, *v_f2_p0, *v_f2_p1, *v_Int_f2_p0;
  

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

  FLOAT dt_half = 0.5 * dt;
  FLOAT dt_mul_dx_inverse = dt / dx;
  FLOAT dt_mul_dy_inverse = dt / dy;
  FLOAT dt_mul_dx_square_inverse = dt / (dx * dx);
  FLOAT dt_mul_dy_square_inverse = dt / (dy * dy);

  // Allocate host memory for the location vector
  v_h_f = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_C1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_h_C_Exact = (FLOAT *)malloc(N_grid * sizeof(FLOAT));

  // Allocate device memory for the location vector
  //*****************position vector******************
  v_x = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_y = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_xhf = (FLOAT *)malloc(N_points * sizeof(FLOAT));
  v_yhf = (FLOAT *)malloc(N_points * sizeof(FLOAT));
  v_xhf_bar1 = (FLOAT *)malloc(N_points * sizeof(FLOAT));
  v_yhf_bar2 = (FLOAT *)malloc(N_points * sizeof(FLOAT));
  v_xhf_bar3 = (FLOAT *)malloc(N_points * sizeof(FLOAT));

  //***************Intersection points***********************
  v_x_inter1 = (FLOAT *)malloc(N_points_inter1 * sizeof(FLOAT));
  v_y_inter2 = (FLOAT *)malloc(N_points_inter2 * sizeof(FLOAT));
  v_x_inter3 = (FLOAT *)malloc(N_points_inter1 * sizeof(FLOAT));

  // Create device vector to store the solution at each time step
  //*********************** step 1 ***********************
  v_C0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_C_xphf = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_C0_Remap = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_DC0_x_bar = (FLOAT *)malloc(N_points_inter1 * sizeof(FLOAT));

  //*********************** step 2 ***********************
  v_IntC_y0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_IntC_yp1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_IntC_y0_Remap = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_DIntC_y0_bar = (FLOAT *)malloc(N_points_inter2 * sizeof(FLOAT));

  //*********************** step 3 ***********************
  v_Cp1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_IntC_xphf = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_DIntC_xphf_bar = (FLOAT *)malloc(N_points_inter1 * sizeof(FLOAT));

  // Create device vector
  v_CL = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_CR = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_C6 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_Dltc = (FLOAT *)malloc(N_grid * sizeof(FLOAT));

  // Create device vector to store matrix elements
  v_coA = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_coB = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_coC = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_b = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  p = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  q = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_Cptb = (FLOAT *)malloc(N_grid * sizeof(FLOAT));

  // Create device vector to store source term
  v_f1_p0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_f1_p1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_Int_f1_p0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_f2_p0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_f2_p1 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));
  v_Int_f2_p0 = (FLOAT *)malloc(N_grid * sizeof(FLOAT));

  clock_t start = clock();
 // Calculte Initial condition and Exact solution on CPU
  clock_t t2 = clock();
  for (i_cell = 0; i_cell < Nx; i_cell++){
    x = X_min + (i_cell + 0.5) * dx;
      for (j_cell = 0; j_cell < Ny; j_cell++){
          y = Y_min + (j_cell + 0.5) * dy;
          tid = i_cell * Ny + j_cell;       
          v_C0[tid] = C_Exact(x, y, 0, Kx);
          v_h_C_Exact[tid] = C_Exact(x, y, T_max, Kx);
      }
  } 
  output_result(v_C0, 0, Nx, Ny, dt);

  //***************************** parallel computation on GPU ****************************

  // Caluculate the location vector on GPU
  Get_Euler_points(v_x, v_y, Nx, Ny, N_grid, dx, dy);

  Get_Lagrange_points
  (v_xhf, v_yhf, v_xhf_bar1, v_yhf_bar2, v_xhf_bar3, Nx, Ny, N_points, dx, dy, dt); 
   
  //********************* step 1 **************************
  Intersection_point_vet_x
  (v_xhf_bar1, v_yhf, v_y, v_x_inter1, Nx, Ny, N_points_inter1); 
  
  //********************* step 2 **************************
  Intersection_point_hori_y
  (v_xhf, v_yhf_bar2, v_x, v_y_inter2, Nx, Ny, N_points_inter2); 
  
  //********************* step 3 **************************
  Intersection_point_vet_x
  (v_xhf_bar3, v_yhf, v_y, v_x_inter3, Nx, Ny, N_points_inter1);   
  
 FLOAT sumf = 0.0;
  for (itime = 1; itime <= Nt; itime++){
    t0 = (itime - 1) * dt;
    t1 = itime * dt;
    //****************************** step 1********************************
    //********** v_C0_Remap = Remap(v_C0) **********
    x_Remap_part1(v_CL, v_CR, v_C6, v_Dltc, v_C0, N_grid, Nx, Ny);
    x_Remap_part2(v_CL, v_CR, v_C6, v_Dltc, v_C0, v_C0_Remap, 
                  v_xhf, v_x_inter1, N_grid, Nx, Ny, dx);

    //*************** Diffusion term ********************
    x_Diff(v_CL, v_CR, v_C0, v_DC0_x_bar, v_xhf, v_x_inter1, 
            Kx, N_points_inter1, Nx, Ny, dx);

    // ************ source term at t^(n+1/2) ************
    get_source_term(v_x, v_y, v_f1_p1, t1-0.5*dt, Kx, N_grid);  
    sumf += sum(v_f1_p1, N_grid);

    // *************** source term at t^n ***************
    get_source_term(v_x, v_y, v_f1_p0, t0, Kx, N_grid);  
    x_Remap_part1(v_CL, v_CR, v_C6, v_Dltc, v_f1_p0, N_grid, Nx, Ny);
    x_Remap_part2(v_CL, v_CR, v_C6, v_Dltc, v_f1_p0, v_Int_f1_p0, 
                  v_xhf, v_x_inter1, N_grid, Nx, Ny, dx);
    sumf += sum(v_Int_f1_p0, N_grid);

    //*************** Matrix and vector******************
    x_Create_matrix
    (v_coA, v_coB, v_coC, v_b, v_C0_Remap, v_DC0_x_bar, v_Int_f1_p0, 
    v_f1_p1, N_grid, Nx, Kx, dt, dt_mul_dx_inverse, dt_mul_dx_square_inverse);
    
    //****************** Compact Scheme ******************
    x_Compact_operater(v_b, v_Cptb, N_grid, Ny, Nx);
  
    //********* get the solution of x-direction **********
    x_Thomas(v_coA, v_coB, v_coC, v_Cptb, v_C_xphf, p, q, N_grid, Nx, Ny);
    
    //******************************** step 2 ***********************************
    //************* v_Int_C = x_Remap(v_C) **********
    y_Remap_part1(v_CL, v_CR, v_C6, v_Dltc, v_C_xphf, N_grid, Nx, Ny);
    y_Remap_part2
    (v_CL, v_CR, v_C6, v_Dltc, v_C_xphf, v_IntC_y0,
    v_yhf, v_y_inter2, N_grid, Nx, Ny, dy);

    //**************** Diffusion term ********************
    y_Diff
    (v_CL, v_CR, v_C_xphf, v_DIntC_y0_bar, v_yhf, v_y_inter2, 
    Ky, N_points_inter2, Nx, Ny, dy, dt_mul_dy_inverse);

    // ************ source term at t^(n+1) ***************
    get_source_term(v_x, v_y, v_f2_p1, t1, Ky, N_grid);  
    sumf += sum(v_f2_p1, N_grid);
    
    // *************** source term at t^n ****************
    get_source_term(v_x, v_y, v_f2_p0, t0, Ky, N_grid);  
    y_Remap_part1(v_CL, v_CR, v_C6, v_Dltc, v_f2_p0, N_grid, Nx, Ny);
    y_Remap_part2
    (v_CL, v_CR, v_C6, v_Dltc, v_f2_p0, v_Int_f2_p0,
    v_yhf, v_y_inter2, N_grid, Nx, Ny, dy);
    sumf += sum(v_Int_f2_p0, N_grid);

    //*************** Matrix and vector******************
    y_Create_matrix
    (v_coA, v_coB, v_coC, v_b, v_IntC_y0, v_DIntC_y0_bar, v_f2_p1, 
    v_Int_f2_p0, N_grid, Ny, Ky, dt, dt_mul_dy_inverse, dt_mul_dy_square_inverse);
    
    
    //***************** Compact Scheme ******************
    y_Compact_operater(v_b, v_Cptb, N_grid, Ny);
    
   
    //********* get the solution of y-direction *********
    y_Thomas(v_coA, v_coB, v_coC, v_Cptb, v_IntC_yp1, p, q, N_grid, Nx, Ny);
    

    //*********************************** step 3 **************************************
    //*************** caluculate CL, CR *****************
    x_Remap_part1(v_CL, v_CR, v_C6, v_Dltc, v_IntC_yp1, N_grid, Nx, Ny);
    x_Remap_part2(v_CL, v_CR, v_C6, v_Dltc, v_IntC_yp1, v_IntC_xphf, 
    v_xhf, v_x_inter1, N_grid, Nx, Ny, dx);

    //****************** Diffusion term *****************
    x_Diff
    (v_CL, v_CR, v_IntC_xphf, v_DIntC_xphf_bar, v_xhf, 
    v_x_inter3, Kx, N_points_inter1, Nx, Ny, dx);

    // ************** source term at t^(n+1) ************
    get_source_term(v_x, v_y, v_f1_p1, t1, Kx, N_grid);  
    sumf += sum(v_f1_p1, N_grid);
    
    // ************ source term at t^(n+1/2) *************
    x_Remap_part1(v_CL, v_CR, v_C6, v_Dltc, v_f1_p0, N_grid, Nx, Ny);
    x_Remap_part2
    (v_CL, v_CR, v_C6, v_Dltc, v_f1_p0, v_Int_f1_p0, 
    v_xhf, v_x_inter1, N_grid, Nx, Ny, dx);
    sumf += sum(v_Int_f1_p0, N_grid);   

    //*************** Matrix and vector******************
    x_Create_matrix
    (v_coA, v_coB, v_coC, v_b, v_IntC_xphf, v_DIntC_xphf_bar, v_Int_f1_p0, 
    v_f1_p1, N_grid, Nx, Kx, dt, dt_mul_dx_inverse, dt_mul_dx_square_inverse);
    

    //**************** Compact Scheme ********************
    x_Compact_operater(v_b, v_Cptb, N_grid, Ny, Nx);
    

    //********* get the solution of x-direction ***********
    x_Thomas(v_coA, v_coB, v_coC, v_Cptb, v_Cp1, p, q, N_grid, Nx, Ny);
    
    
    //************************* Update the solution *******************************
    v_C0 = v_Cp1;
  
  }
  v_h_C1 = v_C0;

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
    cmass_initial = cmass_initial + v_C0[tid];
  }
  cmass_end = cmass_end * dx * dy;
  cmass_initial = cmass_initial * dx * dy;
  Cmassdif = fabs(cmass_initial - cmass_end - sumf * dx * dy);
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
  printf("cmass_end:%.15e\n", cmass_end);
  printf("cmass_initial:%.15e\n", cmass_initial);
  
  double time = (end - start) / (double)CLOCKS_PER_SEC;
  cout << "time:" << time << endl;


  // Free device memory
  free(v_x);
  free(v_y);
  free(v_xhf);
  free(v_yhf);
  free(v_xhf_bar1);
  free(v_yhf_bar2);
  free(v_xhf_bar3);
  free(v_x_inter1);
  free(v_y_inter2);
  free(v_x_inter3);

  free(v_C0);
  free(v_C_xphf);
  free(v_C0_Remap);
  free(v_DC0_x_bar);
  free(v_IntC_y0);
  free(v_IntC_yp1);
  free(v_IntC_y0_Remap);
  free(v_DIntC_y0_bar);
  free(v_Cp1);
  free(v_IntC_xphf);
  free(v_DIntC_xphf_bar);

  free(v_CL);
  free(v_CR);
  free(v_C6);
  free(v_Dltc);
  free(v_coA);
  free(v_coB);
  free(v_coC);
  free(v_Cptb);
  free(v_b);
  free(p);
  free(q);
  free(v_f1_p0);
  free(v_f1_p1);
  free(v_Int_f1_p0);
  free(v_f2_p0);
  free(v_f2_p1);
  free(v_Int_f2_p0);

  // Free host memory
  free(v_h_f);
  free(v_h_C1);
  free(v_h_C_Exact);


  return 0;
}