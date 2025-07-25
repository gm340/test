#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"device_functions.h"
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include"segy.h"
#include "functions.h"

#ifndef MAX 
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#endif

#define PI 3.141592653
#define N 6
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

__device__ float a[6] = { 1.2213365, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };
__device__ float d0;
__device__ float c1[N + 1] = { 0.0000,0.8571,-0.2679,0.0794,-0.0179,0.0026,-0.0002 };

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), file, line);
		exit(int(err));
	}
}

float* initializeArray(int row, int col, float value)
{
	float* array = (float*)malloc(row * col * sizeof(float));

	if (array == NULL) {

		printf("Memory allocation failed. Exiting...\n");
		exit(1);
	}

	for (int i = 0; i < row * col; i++) {
		array[i] = value;
	}
	return array;
}

void** alloc2(size_t n1, size_t n2, size_t size)
{
	size_t i2;
	void** p;

	if ((p = (void**)malloc(n2 * sizeof(void*))) == NULL)
		return NULL;
	if ((p[0] = (void*)malloc(n2 * n1 * size)) == NULL) {
		free(p);
		return NULL;
	}
	for (i2 = 0; i2 < n2; i2++)
		p[i2] = (char*)p[0] + size * n1 * i2;
	return p;
}

int** alloc2int(size_t n1, size_t n2)
{
	return (int**)alloc2(n1, n2, sizeof(int));
}

void laplace_filter(int adj, int Zn1, int Xn1, float* in, float* out)
{
	int iz, ix, j;
	for (j = 0; j < Xn1 * Zn1; j++)
		out[j] = 0.0;

	for (ix = 0; ix < Xn1; ix++) {
		for (iz = 0; iz < Zn1; iz++) {
			j = iz + ix * Zn1;
			if (iz > 0) {
				if (adj) {
					out[j - 1] -= in[j];
					out[j] += in[j];
				}
				else {
					out[j] += in[j] - in[j - 1];
				}
			}
			if (iz < Zn1 - 1) {
				if (adj) {
					out[j + 1] -= in[j];
					out[j] += in[j];
				}
				else {
					out[j] += in[j] - in[j + 1];
				}
			}
			if (ix > 0) {
				if (adj) {
					out[j - Zn1] -= in[j];
					out[j] += in[j];
				}
				else {
					out[j] += in[j] - in[j - Zn1];
				}
			}
			if (ix < Xn1 - 1) {
				if (adj) {
					out[j + Zn1] -= in[j];
					out[j] += in[j];
				}
				else {
					out[j] += in[j] - in[j + Zn1];
				}
			}
		}
	}

}

void swap_pointer(float*& p1, float*& p2)
{
	float* temp = p1;
	p1 = p2;
	p2 = temp;
}

void index_shot_update(int min_shot, int max_shot, int** table, float dis_shot, float disx)
{
	for (int i = 0; i <= max_shot - min_shot; i++)
	{
		table[i][0] = i + 1;
		table[i][1] = ceil(dis_shot * (i + min_shot));
		table[i][2] = 0;
		table[i][3] = disx;
	}

}

int index_shot(char* fn, int* nt, float* dt, int* ns, int** table)
{
	bhed   Bh;
	segy  Th;

	FILE* fp;
	int    cx_min_s, cx_max_s;
	int    ntr, pos;

	fp = fopen(fn, "rb");
	if (fp == NULL) {
		printf("Sorry,cann't open seismic file!\n");
		return 1;
	}

	fseek(fp, 0, SEEK_SET);
	fread(&Th, 240, 1, fp); 
	*nt = (int)Th.ns;
	*dt = Th.dt / 1000000.0;

	int TL = (*nt) * sizeof(float);

	fseek(fp, 0, SEEK_SET);

	*ns = 0;
	pos = 0;
	int sx0 = -999999;
	int sy0 = -999999;
	for (; ; ) {
		fread(&Th, 240, 1, fp);
		if (feof(fp)) {
			int ns0 = *ns;
			table[ns0 - 1][0] = ns0;
			table[ns0 - 1][1] = ntr;
			table[ns0 - 1][2] = sx0;
			table[ns0 - 1][3] = 0; 
			table[ns0 - 1][4] = cx_min_s;
			table[ns0 - 1][5] = 0;
			table[ns0 - 1][6] = cx_max_s;
			table[ns0 - 1][7] = 0;
			table[ns0 - 1][8] = pos - ntr;
			break;
		}

		int sx = Th.sx;
		int sy = 0;//2d,sy=0
		int gx = Th.gx;
		int gy = 0;

		int xmin = MIN(sx, gx);
		int xmax = MAX(sx, gx);

		if (sx != sx0) {
			if (pos > 0) {
				int ns0 = *ns;
				table[ns0 - 1][0] = ns0;
				table[ns0 - 1][1] = ntr;
				table[ns0 - 1][2] = sx0;
				table[ns0 - 1][3] = 0; 
				table[ns0 - 1][4] = cx_min_s;
				table[ns0 - 1][5] = 0;
				table[ns0 - 1][6] = cx_max_s;
				table[ns0 - 1][7] = 0;
				table[ns0 - 1][8] = pos - ntr;
			}
			(*ns)++;
			if ((*ns) % 50 == 0)printf(" %dth shot has been indexed!\n", (*ns));
			ntr = 1;
		
			sx0 = sx;
			sy0 = sy;

			cx_min_s = 999999;
			cx_max_s = -999999;
		}
		else
		{
			ntr++;
		}

		pos++;
		if (xmin < cx_min_s) cx_min_s = xmin;
		if (xmax > cx_max_s) cx_max_s = xmax;

		fseek(fp, TL, SEEK_CUR);
	}
	fclose(fp);

	return 0;
}

void   read_shot_gather_su(char* fn, long long int pos, int ntr, int nt, float* dat, int* gc)
{
	int i;
	FILE* fp;

	fp = fopen(fn, "rb");
	if (fp == NULL) {
		printf("Sorry,cann't open input seismic file!\n");
		exit(0);
	}

	int TL = 240 + nt * sizeof(float);

	fseek(fp, (long long int)TL * pos, SEEK_SET);

	segy Th;

	for (i = 0; i < ntr; i++) {
		fread(&Th, 240, 1, fp);
		gc[i] = Th.gx;
		fread(&dat[i * nt], sizeof(float), nt, fp);
	}

	fclose(fp);
}


void readFile(char FNvelocity[], char FNvs0[], float* vp0, float* vs0, int Xn, int Zn, int L)
{
	int i, j, idx;
	float vmax, vmin;
	float emax, emin;
	FILE* fp1, * fp2;

	if ((fp1 = fopen(FNvelocity, "rb")) == NULL)
	{
		printf("error open <%s>!\n", FNvelocity);
		exit(1);
	}
	if ((fp2 = fopen(FNvs0, "rb")) == NULL)
	{
		printf("error open <%s>!\n", FNvs0);
	}

	vmin = emin = 999999.9;
	vmax = emax = -999999.9;

	for (i = L; i < Xn - L; i++)
		for (j = L; j < Zn - L; j++)
		{
			idx = i * Zn + j;
			fread(&vp0[idx], sizeof(float), 1, fp1);
			vp0[idx] *= 1.0;
			fread(&vs0[idx], sizeof(float), 1, fp2);
			vs0[idx] *= 1.0;

			
			if (vmax < vp0[idx]) vmax = vp0[idx];
			if (vmin > vp0[idx]) vmin = vp0[idx];
			if (emax < vs0[idx]) emax = vs0[idx];
			if (emin > vs0[idx]) emin = vs0[idx];

		}
	fclose(fp1); fclose(fp2);

	printf("Load (vp0, vs0) successfully!\n");

	printf("		Vp Range (%.1f - %.1f)[m/s]\n", vmin, vmax);
	printf("		Vs Range (%.1f - %.1f)[m/s]\n", emin, emax);
}




void ini_bdr(int Xn, int Zn, int L, float* ee)
{
	int ix, iz, idx;
	for (idx = 0; idx < Xn * Zn; idx++)
	{
		ix = idx / Zn;
		iz = idx % Zn;

	
		if (ix < L) {
			ee[idx] = ee[L * Zn + iz];
		}
		else if (ix >= Xn - L) {
			ee[idx] = ee[(Xn - L - 1) * Zn + iz];
		}
	}
	for (idx = 0; idx < Xn * Zn; idx++)
	{
		ix = idx / Zn;
		iz = idx % Zn;

	
		if (iz < L)
		{
			ee[idx] = ee[ix * Zn + L];
		}
		else if (iz >= Zn - L)
		{
			ee[idx] = ee[ix * Zn + Zn - L - 1];
		}
	}
}

__global__ void get_d0(int L, float dx)
{
	d0 = 1.5 * log(1e9) / (L * dx);
}


__global__ void pml_coef(
	int Xn,
	int Zn,
	int L,
	float* vp0,
	float* damp
)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;

	if (ix >= N && ix < Xn - N && iz >= N && iz < Zn - N)
	{
		if (ix < L)
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(L - ix) / L, 2.0);
		}
		else if (ix >= Xn - L) 
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(ix - (Xn - L - 1)) / L, 2.0);
		}
		else if (ix >= L && ix < Xn - L && iz < L)
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(L - iz) / L, 2.0);
		}
		else if (ix >= L && ix < Xn - L && iz >= Zn - L)
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(iz - (Zn - L - 1)) / L, 2.0);
		}
		else 
		{
			damp[idx] = 0.0;
		}
	}

}

__global__ void add_source(
	int Zn,
	int shotx, int shotz,
	int it,
	float dt,
	float FM,
	float* u
)
{
	
	int tdelay = ceil(1.0 / (FM * dt));
	float wavelet = (1.0 - 2.0 * PI * PI * FM * FM * dt * dt * (it - tdelay) * (it - tdelay)) * exp(-PI * PI * FM * FM * dt * dt * (it - tdelay) * (it - tdelay));
	u[shotx * Zn + shotz] += 1.0 * wavelet * 100.0;
}

__global__ void cal_u_z(
	int Xn, int Zn,
	int it,
	float dz,
	float* u,
	float* u_z
)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;

	int m;

	if (ix >= N && ix < Xn - N && iz >= N && iz < Zn - N)
	{
		
		for (m = 0; m <= N; m++)
		{
			u_z[idx] += c1[m] * (u[ix * Zn + iz + m] - u[ix * Zn + iz - m]) / dz;
		}
	}

}

__global__ void cal_u_xz(
	int Xn, int Zn,
	float dx,
	float* u_z,
	float* u_xz
)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;

	int m;

	if (ix >= N && ix < Xn - N && iz >= N && iz < Zn - N)
	{

		for (m = 0; m <= N; m++)
		{
			u_xz[idx] += (c1[m] * (u_z[(ix + m) * Zn + iz] - u_z[(ix - m) * Zn + iz])) / dx;
		}
	}

}



__global__ void update_u(
	int Xn, int Zn, int L,
	float dx, float dz, float dt,
	float* vp0, float* vs0,
	float* damp,
	float* upx0, float* upx1, float* upx2,
	float* upz0, float* upz1, float* upz2,
	float* usx0, float* usx1, float* usx2,
	float* usz0, float* usz1, float* usz2,
	float* ux, float* uz,
	float* theta, float* omega,
	float* duzdx, float* duzdz, float* duxdz, float* duxdx,
	int flag
)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i * Zn + j;

	int m;

	float dthetadx = 0.0f;
	float dthetadz = 0.0f;
	float domegadx = 0.0f;
	float domegadz = 0.0f;

	float duz_xdz = 0.0f;
	float duz_zdx = 0.0f;
	float dux_zdx = 0.0f;
	float dux_xdz = 0.0f;

	if (flag == 1) 
	{
		if (i >= N && i < Xn - N && j >= N && j < Zn - N)
		{

			dthetadx = (a[0] * (theta[(i + 1) * Zn + j] - theta[(i - 0) * Zn + j])
				+ a[1] * (theta[(i + 2) * Zn + j] - theta[(i - 1) * Zn + j])
				+ a[2] * (theta[(i + 3) * Zn + j] - theta[(i - 2) * Zn + j])
				+ a[3] * (theta[(i + 4) * Zn + j] - theta[(i - 3) * Zn + j])
				+ a[4] * (theta[(i + 5) * Zn + j] - theta[(i - 4) * Zn + j])
				+ a[5] * (theta[(i + 6) * Zn + j] - theta[(i - 5) * Zn + j])) / dx;

			dthetadz = (a[0] * (theta[(i)*Zn + j + 1] - theta[(i)*Zn + j - 0])
				+ a[1] * (theta[(i)*Zn + j + 2] - theta[(i)*Zn + j - 1])
				+ a[2] * (theta[(i)*Zn + j + 3] - theta[(i)*Zn + j - 2])
				+ a[3] * (theta[(i)*Zn + j + 4] - theta[(i)*Zn + j - 3])
				+ a[4] * (theta[(i)*Zn + j + 5] - theta[(i)*Zn + j - 4])
				+ a[5] * (theta[(i)*Zn + j + 6] - theta[(i)*Zn + j - 5])) / dz;

			domegadx = (a[0] * (omega[(i + 0) * Zn + j] - omega[(i - 1) * Zn + j])
				+ a[1] * (omega[(i + 1) * Zn + j] - omega[(i - 2) * Zn + j])
				+ a[2] * (omega[(i + 2) * Zn + j] - omega[(i - 3) * Zn + j])
				+ a[3] * (omega[(i + 3) * Zn + j] - omega[(i - 4) * Zn + j])
				+ a[4] * (omega[(i + 4) * Zn + j] - omega[(i - 5) * Zn + j])
				+ a[5] * (omega[(i + 5) * Zn + j] - omega[(i - 6) * Zn + j])) / dx;

			domegadz = (a[0] * (omega[(i)*Zn + j + 0] - omega[(i)*Zn + j - 1])
				+ a[1] * (omega[(i)*Zn + j + 1] - omega[(i)*Zn + j - 2])
				+ a[2] * (omega[(i)*Zn + j + 2] - omega[(i)*Zn + j - 3])
				+ a[3] * (omega[(i)*Zn + j + 3] - omega[(i)*Zn + j - 4])
				+ a[4] * (omega[(i)*Zn + j + 4] - omega[(i)*Zn + j - 5])
				+ a[5] * (omega[(i)*Zn + j + 5] - omega[(i)*Zn + j - 6])) / dz;


			dux_xdz = (a[0] * (duxdx[(i)*Zn + j + 1] - duxdx[(i)*Zn + j - 0])
				+ a[1] * (duxdx[(i)*Zn + j + 2] - duxdx[(i)*Zn + j - 1])
				+ a[2] * (duxdx[(i)*Zn + j + 3] - duxdx[(i)*Zn + j - 2])
				+ a[3] * (duxdx[(i)*Zn + j + 4] - duxdx[(i)*Zn + j - 3])
				+ a[4] * (duxdx[(i)*Zn + j + 5] - duxdx[(i)*Zn + j - 4])
				+ a[5] * (duxdx[(i)*Zn + j + 6] - duxdx[(i)*Zn + j - 5])) / dz;

			dux_zdx = (a[0] * (duxdz[(i + 0) * Zn + j] - duxdz[(i - 1) * Zn + j])
				+ a[1] * (duxdz[(i + 1) * Zn + j] - duxdz[(i - 2) * Zn + j])
				+ a[2] * (duxdz[(i + 2) * Zn + j] - duxdz[(i - 3) * Zn + j])
				+ a[3] * (duxdz[(i + 3) * Zn + j] - duxdz[(i - 4) * Zn + j])
				+ a[4] * (duxdz[(i + 4) * Zn + j] - duxdz[(i - 5) * Zn + j])
				+ a[5] * (duxdz[(i + 5) * Zn + j] - duxdz[(i - 6) * Zn + j])) / dx;

			duz_xdz = (a[0] * (duzdx[(i)*Zn + j + 0] - duzdx[(i)*Zn + j - 1])
				+ a[1] * (duzdx[(i)*Zn + j + 1] - duzdx[(i)*Zn + j - 2])
				+ a[2] * (duzdx[(i)*Zn + j + 2] - duzdx[(i)*Zn + j - 3])
				+ a[3] * (duzdx[(i)*Zn + j + 3] - duzdx[(i)*Zn + j - 4])
				+ a[4] * (duzdx[(i)*Zn + j + 4] - duzdx[(i)*Zn + j - 5])
				+ a[5] * (duzdx[(i)*Zn + j + 5] - duzdx[(i)*Zn + j - 6])) / dz;

			duz_zdx = (a[0] * (duzdz[(i + 1) * Zn + j] - duzdz[(i - 0) * Zn + j])
				+ a[1] * (duzdz[(i + 2) * Zn + j] - duzdz[(i - 1) * Zn + j])
				+ a[2] * (duzdz[(i + 3) * Zn + j] - duzdz[(i - 2) * Zn + j])
				+ a[3] * (duzdz[(i + 4) * Zn + j] - duzdz[(i - 3) * Zn + j])
				+ a[4] * (duzdz[(i + 5) * Zn + j] - duzdz[(i - 4) * Zn + j])
				+ a[5] * (duzdz[(i + 6) * Zn + j] - duzdz[(i - 5) * Zn + j])) / dx;


			upx2[idx] = (2.0 - damp[idx] * dt) * upx1[idx] - (1.0 - damp[idx] * dt) * upx0[idx] + dt * dt * dthetadx;
			upz2[idx] = (2.0 - damp[idx] * dt) * upz1[idx] - (1.0 - damp[idx] * dt) * upz0[idx] + dt * dt * dthetadz;

			usx2[idx] = (2.0 - damp[idx] * dt) * usx1[idx] - (1.0 - damp[idx] * dt) * usx0[idx] + dt * dt * (domegadz + 2 * (duz_xdz - duz_zdx));
			usz2[idx] = (2.0 - damp[idx] * dt) * usz1[idx] - (1.0 - damp[idx] * dt) * usz0[idx] + dt * dt * (-domegadx + 2 * (dux_zdx - dux_xdz));

			ux[idx] = upx2[idx] + usx2[idx];
			uz[idx] = upz2[idx] + usz2[idx];

		}
	}

	if (flag == 2)
	{
		if (i >= L && i < Xn - L && j >= L && j < Zn - L)
		{

			dthetadx = (a[0] * (theta[(i + 1) * Zn + j] - theta[(i - 0) * Zn + j])
				+ a[1] * (theta[(i + 2) * Zn + j] - theta[(i - 1) * Zn + j])
				+ a[2] * (theta[(i + 3) * Zn + j] - theta[(i - 2) * Zn + j])
				+ a[3] * (theta[(i + 4) * Zn + j] - theta[(i - 3) * Zn + j])
				+ a[4] * (theta[(i + 5) * Zn + j] - theta[(i - 4) * Zn + j])
				+ a[5] * (theta[(i + 6) * Zn + j] - theta[(i - 5) * Zn + j])) / dx;

			dthetadz = (a[0] * (theta[(i)*Zn + j + 1] - theta[(i)*Zn + j - 0])
				+ a[1] * (theta[(i)*Zn + j + 2] - theta[(i)*Zn + j - 1])
				+ a[2] * (theta[(i)*Zn + j + 3] - theta[(i)*Zn + j - 2])
				+ a[3] * (theta[(i)*Zn + j + 4] - theta[(i)*Zn + j - 3])
				+ a[4] * (theta[(i)*Zn + j + 5] - theta[(i)*Zn + j - 4])
				+ a[5] * (theta[(i)*Zn + j + 6] - theta[(i)*Zn + j - 5])) / dz;

			domegadx = (a[0] * (omega[(i + 0) * Zn + j] - omega[(i - 1) * Zn + j])
				+ a[1] * (omega[(i + 1) * Zn + j] - omega[(i - 2) * Zn + j])
				+ a[2] * (omega[(i + 2) * Zn + j] - omega[(i - 3) * Zn + j])
				+ a[3] * (omega[(i + 3) * Zn + j] - omega[(i - 4) * Zn + j])
				+ a[4] * (omega[(i + 4) * Zn + j] - omega[(i - 5) * Zn + j])
				+ a[5] * (omega[(i + 5) * Zn + j] - omega[(i - 6) * Zn + j])) / dx;

			domegadz = (a[0] * (omega[(i)*Zn + j + 0] - omega[(i)*Zn + j - 1])
				+ a[1] * (omega[(i)*Zn + j + 1] - omega[(i)*Zn + j - 2])
				+ a[2] * (omega[(i)*Zn + j + 2] - omega[(i)*Zn + j - 3])
				+ a[3] * (omega[(i)*Zn + j + 3] - omega[(i)*Zn + j - 4])
				+ a[4] * (omega[(i)*Zn + j + 4] - omega[(i)*Zn + j - 5])
				+ a[5] * (omega[(i)*Zn + j + 5] - omega[(i)*Zn + j - 6])) / dz;


			dux_xdz = (a[0] * (duxdx[(i)*Zn + j + 1] - duxdx[(i)*Zn + j - 0])
				+ a[1] * (duxdx[(i)*Zn + j + 2] - duxdx[(i)*Zn + j - 1])
				+ a[2] * (duxdx[(i)*Zn + j + 3] - duxdx[(i)*Zn + j - 2])
				+ a[3] * (duxdx[(i)*Zn + j + 4] - duxdx[(i)*Zn + j - 3])
				+ a[4] * (duxdx[(i)*Zn + j + 5] - duxdx[(i)*Zn + j - 4])
				+ a[5] * (duxdx[(i)*Zn + j + 6] - duxdx[(i)*Zn + j - 5])) / dz;

			dux_zdx = (a[0] * (duxdz[(i + 0) * Zn + j] - duxdz[(i - 1) * Zn + j])
				+ a[1] * (duxdz[(i + 1) * Zn + j] - duxdz[(i - 2) * Zn + j])
				+ a[2] * (duxdz[(i + 2) * Zn + j] - duxdz[(i - 3) * Zn + j])
				+ a[3] * (duxdz[(i + 3) * Zn + j] - duxdz[(i - 4) * Zn + j])
				+ a[4] * (duxdz[(i + 4) * Zn + j] - duxdz[(i - 5) * Zn + j])
				+ a[5] * (duxdz[(i + 5) * Zn + j] - duxdz[(i - 6) * Zn + j])) / dx;

			duz_xdz = (a[0] * (duzdx[(i)*Zn + j + 0] - duzdx[(i)*Zn + j - 1])
				+ a[1] * (duzdx[(i)*Zn + j + 1] - duzdx[(i)*Zn + j - 2])
				+ a[2] * (duzdx[(i)*Zn + j + 2] - duzdx[(i)*Zn + j - 3])
				+ a[3] * (duzdx[(i)*Zn + j + 3] - duzdx[(i)*Zn + j - 4])
				+ a[4] * (duzdx[(i)*Zn + j + 4] - duzdx[(i)*Zn + j - 5])
				+ a[5] * (duzdx[(i)*Zn + j + 5] - duzdx[(i)*Zn + j - 6])) / dz;

			duz_zdx = (a[0] * (duzdz[(i + 1) * Zn + j] - duzdz[(i - 0) * Zn + j])
				+ a[1] * (duzdz[(i + 2) * Zn + j] - duzdz[(i - 1) * Zn + j])
				+ a[2] * (duzdz[(i + 3) * Zn + j] - duzdz[(i - 2) * Zn + j])
				+ a[3] * (duzdz[(i + 4) * Zn + j] - duzdz[(i - 3) * Zn + j])
				+ a[4] * (duzdz[(i + 5) * Zn + j] - duzdz[(i - 4) * Zn + j])
				+ a[5] * (duzdz[(i + 6) * Zn + j] - duzdz[(i - 5) * Zn + j])) / dx;


			upx0[idx] = 2.0 * upx1[idx] - upx2[idx] + dt * dt * dthetadx;
			upz0[idx] = 2.0 * upz1[idx] - upz2[idx] + dt * dt * dthetadz;

			usx0[idx] = 2.0 * usx1[idx] - usx2[idx] + dt * dt * (domegadz + 2 * (duz_xdz - duz_zdx));
			usz0[idx] = 2.0 * usz1[idx] - usz2[idx] + dt * dt * (-domegadx + 2 * (dux_zdx - dux_xdz));

			ux[idx] = upx0[idx] + usx0[idx];
			uz[idx] = upz0[idx] + usz0[idx];

		}
	}

	if (flag == 3) 
	{
		if (i >= N && i < Xn - N && j >= N && j < Zn - N)
		{

			dthetadx = (a[0] * (theta[(i + 1) * Zn + j] - theta[(i - 0) * Zn + j])
				+ a[1] * (theta[(i + 2) * Zn + j] - theta[(i - 1) * Zn + j])
				+ a[2] * (theta[(i + 3) * Zn + j] - theta[(i - 2) * Zn + j])
				+ a[3] * (theta[(i + 4) * Zn + j] - theta[(i - 3) * Zn + j])
				+ a[4] * (theta[(i + 5) * Zn + j] - theta[(i - 4) * Zn + j])
				+ a[5] * (theta[(i + 6) * Zn + j] - theta[(i - 5) * Zn + j])) / dx;

			dthetadz = (a[0] * (theta[(i)*Zn + j + 1] - theta[(i)*Zn + j - 0])
				+ a[1] * (theta[(i)*Zn + j + 2] - theta[(i)*Zn + j - 1])
				+ a[2] * (theta[(i)*Zn + j + 3] - theta[(i)*Zn + j - 2])
				+ a[3] * (theta[(i)*Zn + j + 4] - theta[(i)*Zn + j - 3])
				+ a[4] * (theta[(i)*Zn + j + 5] - theta[(i)*Zn + j - 4])
				+ a[5] * (theta[(i)*Zn + j + 6] - theta[(i)*Zn + j - 5])) / dz;

			domegadx = (a[0] * (omega[(i + 0) * Zn + j] - omega[(i - 1) * Zn + j])
				+ a[1] * (omega[(i + 1) * Zn + j] - omega[(i - 2) * Zn + j])
				+ a[2] * (omega[(i + 2) * Zn + j] - omega[(i - 3) * Zn + j])
				+ a[3] * (omega[(i + 3) * Zn + j] - omega[(i - 4) * Zn + j])
				+ a[4] * (omega[(i + 4) * Zn + j] - omega[(i - 5) * Zn + j])
				+ a[5] * (omega[(i + 5) * Zn + j] - omega[(i - 6) * Zn + j])) / dx;

			domegadz = (a[0] * (omega[(i)*Zn + j + 0] - omega[(i)*Zn + j - 1])
				+ a[1] * (omega[(i)*Zn + j + 1] - omega[(i)*Zn + j - 2])
				+ a[2] * (omega[(i)*Zn + j + 2] - omega[(i)*Zn + j - 3])
				+ a[3] * (omega[(i)*Zn + j + 3] - omega[(i)*Zn + j - 4])
				+ a[4] * (omega[(i)*Zn + j + 4] - omega[(i)*Zn + j - 5])
				+ a[5] * (omega[(i)*Zn + j + 5] - omega[(i)*Zn + j - 6])) / dz;


			dux_xdz = (a[0] * (duxdx[(i)*Zn + j + 1] - duxdx[(i)*Zn + j - 0])
				+ a[1] * (duxdx[(i)*Zn + j + 2] - duxdx[(i)*Zn + j - 1])
				+ a[2] * (duxdx[(i)*Zn + j + 3] - duxdx[(i)*Zn + j - 2])
				+ a[3] * (duxdx[(i)*Zn + j + 4] - duxdx[(i)*Zn + j - 3])
				+ a[4] * (duxdx[(i)*Zn + j + 5] - duxdx[(i)*Zn + j - 4])
				+ a[5] * (duxdx[(i)*Zn + j + 6] - duxdx[(i)*Zn + j - 5])) / dz;

			dux_zdx = (a[0] * (duxdz[(i + 0) * Zn + j] - duxdz[(i - 1) * Zn + j])
				+ a[1] * (duxdz[(i + 1) * Zn + j] - duxdz[(i - 2) * Zn + j])
				+ a[2] * (duxdz[(i + 2) * Zn + j] - duxdz[(i - 3) * Zn + j])
				+ a[3] * (duxdz[(i + 3) * Zn + j] - duxdz[(i - 4) * Zn + j])
				+ a[4] * (duxdz[(i + 4) * Zn + j] - duxdz[(i - 5) * Zn + j])
				+ a[5] * (duxdz[(i + 5) * Zn + j] - duxdz[(i - 6) * Zn + j])) / dx;

			duz_xdz = (a[0] * (duzdx[(i)*Zn + j + 0] - duzdx[(i)*Zn + j - 1])
				+ a[1] * (duzdx[(i)*Zn + j + 1] - duzdx[(i)*Zn + j - 2])
				+ a[2] * (duzdx[(i)*Zn + j + 2] - duzdx[(i)*Zn + j - 3])
				+ a[3] * (duzdx[(i)*Zn + j + 3] - duzdx[(i)*Zn + j - 4])
				+ a[4] * (duzdx[(i)*Zn + j + 4] - duzdx[(i)*Zn + j - 5])
				+ a[5] * (duzdx[(i)*Zn + j + 5] - duzdx[(i)*Zn + j - 6])) / dz;

			duz_zdx = (a[0] * (duzdz[(i + 1) * Zn + j] - duzdz[(i - 0) * Zn + j])
				+ a[1] * (duzdz[(i + 2) * Zn + j] - duzdz[(i - 1) * Zn + j])
				+ a[2] * (duzdz[(i + 3) * Zn + j] - duzdz[(i - 2) * Zn + j])
				+ a[3] * (duzdz[(i + 4) * Zn + j] - duzdz[(i - 3) * Zn + j])
				+ a[4] * (duzdz[(i + 5) * Zn + j] - duzdz[(i - 4) * Zn + j])
				+ a[5] * (duzdz[(i + 6) * Zn + j] - duzdz[(i - 5) * Zn + j])) / dx;


			upx2[idx] = (2.0 - damp[idx] * dt) * upx1[idx] - (1.0 - damp[idx] * dt) * upx0[idx] + dt * dt * dthetadx;
			upz2[idx] = (2.0 - damp[idx] * dt) * upz1[idx] - (1.0 - damp[idx] * dt) * upz0[idx] + dt * dt * dthetadz;

			usx2[idx] = (2.0 - damp[idx] * dt) * usx1[idx] - (1.0 - damp[idx] * dt) * usx0[idx] + dt * dt * (domegadz + 2 * (duz_xdz - duz_zdx));
			usz2[idx] = (2.0 - damp[idx] * dt) * usz1[idx] - (1.0 - damp[idx] * dt) * usz0[idx] + dt * dt * (-domegadx + 2 * (dux_zdx - dux_xdz));

			ux[idx] = upx2[idx] + usx2[idx];
			uz[idx] = upz2[idx] + usz2[idx];

		}
	}

}

__global__ void update_stress(
	int Xn, int Zn, int L,
	float dx, float dz, float dt,
	float* vp0, float* vs0,
	float* damp,
	float* ux, float* uz,
	float* theta, float* omega,
	float* duzdx, float* duzdz, float* duxdz, float* duxdx,
	int flag
)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i * Zn + j;

	int m;

	float dux_dx = 0.0f;
	float duz_dz = 0.0f;
	float dux_dz = 0.0f;
	float duz_dx = 0.0f;

	if (flag == 1) 
	{
		if (i >= N && i < Xn - N && j >= N && j < Zn - N)
		{

			dux_dx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
				+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
				+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
				+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
				+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
				+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

			dux_dz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
				+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
				+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
				+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
				+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
				+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

			duz_dz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
				+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
				+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
				+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
				+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
				+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

			duz_dx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
				+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
				+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
				+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
				+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
				+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;

			theta[idx] = vp0[idx] * vp0[idx] * (dux_dx + duz_dz);
			omega[idx] = vs0[idx] * vs0[idx] * (dux_dz - duz_dx);

			duzdx[idx] = vs0[idx] * vs0[idx] * duz_dx;
			duzdz[idx] = vs0[idx] * vs0[idx] * duz_dz;
			duxdx[idx] = vs0[idx] * vs0[idx] * dux_dx;
			duxdz[idx] = vs0[idx] * vs0[idx] * dux_dz;
		}
	}


	if (flag == 2) 
	{
		if (i >= L && i < Xn - L && j >= L && j < Zn - L)
		{

			dux_dx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
				+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
				+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
				+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
				+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
				+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

			dux_dz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
				+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
				+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
				+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
				+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
				+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

			duz_dz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
				+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
				+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
				+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
				+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
				+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

			duz_dx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
				+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
				+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
				+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
				+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
				+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;

			theta[idx] = vp0[idx] * vp0[idx] * (dux_dx + duz_dz);
			omega[idx] = vs0[idx] * vs0[idx] * (dux_dz - duz_dx);

			duzdx[idx] = vs0[idx] * vs0[idx] * duz_dx;
			duzdz[idx] = vs0[idx] * vs0[idx] * duz_dz;
			duxdx[idx] = vs0[idx] * vs0[idx] * dux_dx;
			duxdz[idx] = vs0[idx] * vs0[idx] * dux_dz;
		}
	}

	if (flag == 3) 
	{
		if (i >= N && i < Xn - N && j >= N && j < Zn - N)
		{

			dux_dx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
				+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
				+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
				+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
				+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
				+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

			dux_dz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
				+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
				+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
				+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
				+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
				+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

			duz_dz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
				+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
				+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
				+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
				+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
				+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

			duz_dx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
				+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
				+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
				+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
				+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
				+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;

			theta[idx] = vp0[idx] * vp0[idx] * (dux_dx + duz_dz);
			omega[idx] = vs0[idx] * vs0[idx] * (dux_dz - duz_dx);

			duzdx[idx] = vs0[idx] * vs0[idx] * duz_dx;
			duzdz[idx] = vs0[idx] * vs0[idx] * duz_dz;
			duxdx[idx] = vs0[idx] * vs0[idx] * dux_dx;
			duxdz[idx] = vs0[idx] * vs0[idx] * dux_dz;
		}
	}

}

__global__ void cal_illumination(int Xn, int Zn,
	int L,
	float* upx,
	float* upz,
	float* illumination,
	int it,
	int Tn)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;
	int Zn1 = Zn - 2 * L;

	if (ix >= L && ix < Xn - L && iz >= L && iz < Zn - L)
	{
		illumination[(ix - L) * Zn1 + iz - L] += upx[idx] * upx[idx] + upz[idx] * upz[idx];

		if (it == Tn - 1 && illumination[(ix - L) * Zn1 + iz - L] <= 0.0)
		{
			illumination[(ix - L) * Zn1 + iz - L] = 1.0;
		}
	}

}

__global__ void cal_migration(int Xn, int Zn,
	int L,
	float* upx,
	float* upz,
	float* r_upx,
	float* r_upz,
	float* migration)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;
	int Zn1 = Zn - 2 * L;

	if (ix >= L && ix < Xn - L && iz >= L && iz < Zn - L)
	{
		migration[(ix - L) * Zn1 + iz - L] += upx[idx] * r_upx[idx] + upz[idx] * r_upz[idx];
	}
}


__global__ void cal_migration(int Xn, int Zn,
	int L,
	float* upx_down, float* upz_down, float* upx_up, float* upz_up, float* r_upx_up, float* r_upz_up, float* r_upx_down, float* r_upz_down, float* r_usx_up, float* r_usz_up, float* r_usx_down, float* r_usz_down,
	float* upx_right, float* upz_right, float* r_upx_left, float* r_upz_left, float* r_usx_left, float* r_usz_left, float* upx_left, float* upz_left, float* r_upx_right, float* r_upz_right, float* r_usx_right, float* r_usz_right,
	float* migration_poynting_pp,
	float* migration_poynting_ps)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;
	int Zn1 = Zn - 2 * L;

	if (ix >= L && ix < Xn - L && iz >= L && iz < Zn - L)
	{
		migration_poynting_pp[(ix - L) * Zn1 + iz - L] += upx_down[idx] * r_upx_up[idx] + upx_up[idx] * r_upx_down[idx] + upz_down[idx] * r_upz_up[idx] + upz_up[idx] * r_upz_down[idx] + upx_right[idx] * r_upx_left[idx] + upz_right[idx] * r_upz_left[idx] + upx_left[idx] * r_upx_right[idx] + upz_left[idx] * r_upz_right[idx];
		migration_poynting_ps[(ix - L) * Zn1 + iz - L] += upx_down[idx] * r_usx_up[idx] + upx_up[idx] * r_usx_down[idx] + upz_down[idx] * r_usz_up[idx] + upz_up[idx] * r_usz_down[idx] + upx_right[idx] * r_usx_left[idx] + upz_right[idx] * r_usz_left[idx] + upx_left[idx] * r_usx_right[idx] + upz_left[idx] * r_usz_right[idx];
	}
}

__global__ void cal_migration_model(int Xn, int Zn,
	int L,
	float* u,
	float* r_u,
	float* vp0,
	float* epsilon,
	float* delta,
	float* migration)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = ix * Zn + iz;
	int Zn1 = Zn - 2 * L;

	if (ix >= L && ix < Xn - L && iz >= L && iz < Zn - L)
	{
		migration[(ix - L) * Zn1 + iz - L] += 1 / (vp0[idx] * vp0[idx]) * (1 + 2 * epsilon[idx]) * sqrt(2 * delta[idx] + 1) * u[idx] * r_u[idx]; 
	}
}

__global__ void migration_illum(int Xn, int Zn,
	int L,
	float* illumination,
	float* migration)
{
	int iz = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	int Zn1 = Zn - 2 * L;

	if (ix >= L && ix < Xn - L && iz >= L && iz < Zn - L)
	{
		migration[(ix - L) * Zn1 + iz - L] /= illumination[(ix - L) * Zn1 + iz - L];
	}

}

__global__ void shot_record(int Xn, int Zn,
	int Tn, int it,
	int L,
	int recdep,
	float* u, float* record,
	bool symbol)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;

	if (ix < Xn - 2 * L)
	{
	
		if (symbol)
		{
			record[ix * Tn + it] = u[(ix + L) * Zn + recdep];
		}
	
		else
		{
			u[(ix + L) * Zn + recdep] = record[ix * Tn + it];
		}
	}
}

__global__ void wavefield_bdr(int Xn, int Zn,
	int L,
	int it,
	float* u_bdr,
	float* u,
	bool symbol)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int Xn1 = Xn - 2 * L;
	int Zn1 = Zn - 2 * L;
	int size = 2 * N * Xn1 + 2 * N * Zn1;
	int ix, iz;
	if (idx < size)
	{
	
		if (symbol)
		{
			if (idx < N * Xn1) 
			{
				ix = idx % Xn1;
				iz = idx / Xn1 - N;
				u_bdr[it * size + idx] = u[(ix + L) * Zn + iz + L];
			}
			else if (idx >= N * Xn1 && idx < 2 * N * Xn1)
			{
				ix = (idx - N * Xn1) % Xn1;
				iz = (idx - N * Xn1) / Xn1;
				u_bdr[it * size + idx] = u[(ix + L) * Zn + iz + Zn - L];
			}
			else if (idx >= 2 * N * Xn1 && idx < 2 * N * Xn1 + N * Zn1)
			{
				ix = (idx - 2 * N * Xn1) / Zn1 - N;
				iz = (idx - 2 * N * Xn1) % Zn1;
				u_bdr[it * size + idx] = u[(ix + L) * Zn + iz + L];
			}
			else if (idx >= 2 * N * Xn1 + N * Zn1)  
			{
				ix = (idx - 2 * N * Xn1 - N * Zn1) / Zn1;
				iz = (idx - 2 * N * Xn1 - N * Zn1) % Zn1;
				u_bdr[it * size + idx] = u[(ix + Xn - L) * Zn + iz + L];
			}
		}
		
		else
		{
			if (idx < N * Xn1)
			{
				ix = idx % Xn1;
				iz = idx / Xn1 - N;
				u[(ix + L) * Zn + iz + L] = u_bdr[it * size + idx];
			}
			else if (idx >= N * Xn1 && idx < 2 * N * Xn1) 
			{
				ix = (idx - N * Xn1) % Xn1;
				iz = (idx - N * Xn1) / Xn1;
				u[(ix + L) * Zn + iz + Zn - L] = u_bdr[it * size + idx];
			}
			else if (idx >= 2 * N * Xn1 && idx < 2 * N * Xn1 + N * Zn1) 
			{
				ix = (idx - 2 * N * Xn1) / Zn1 - N;
				iz = (idx - 2 * N * Xn1) % Zn1;
				u[(ix + L) * Zn + iz + L] = u_bdr[it * size + idx];
			}
			else if (idx >= 2 * N * Xn1 + N * Zn1)
			{
				ix = (idx - 2 * N * Xn1 - N * Zn1) / Zn1;
				iz = (idx - 2 * N * Xn1 - N * Zn1) % Zn1;
				u[(ix + Xn - L) * Zn + iz + L] = u_bdr[it * size + idx];
			}
		}

	}
}


__global__ void poynting(int Xn, int Zn, int L, float* r_upx2, float* r_upz2, float* r_usx2, float* r_usz2, float* r_upx1, float* r_upz1, float* r_usx1, float* r_usz1,
	float* upx0, float* upz0, float* upx1, float* upz1, float* theta,
	float* r_theta, float* r_omega,	
	float* r_upx_up, float* r_upx_down, float* r_upx_left, float* r_upx_right, float* r_upz_up, float* r_upz_down, float* r_upz_left, float* r_upz_right,
	float* r_usx_up, float* r_usx_down, float* r_usx_left, float* r_usx_right, float* r_usz_up, float* r_usz_down, float* r_usz_left, float* r_usz_right,
	float* upx_up, float* upx_down, float* upx_left, float* upx_right, float* upz_up, float* upz_down, float* upz_left, float* upz_right, float* r_duxdz, float* r_duzdz, float* r_duxdx, float* r_duzdx, int dt)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i * Zn + j;

	float2 Ep_s, Ep_r, Es_r;
	float vpx = 0.0f;
	float vpz = 0.0f;

	float dupx_r = 0.0f;
	float dupz_r = 0.0f;
	float dusx_r = 0.0f;
	float dusz_r = 0.0f;	
	
	if (i >= L && i < Xn - L && j >= L && j < Zn - L)
	{

		vpx = (upx0[i * Zn + j] - upx1[i * Zn + j]) / dt;
		vpz = (upz0[i * Zn + j] - upz1[i * Zn + j]) / dt;
		
		Ep_s.x = -theta[i * Zn + j] * vpx;
		Ep_s.y = -theta[i * Zn + j] * vpz; 

		dupx_r = (r_upx2[i * Zn + j] - r_upx1[i * Zn + j]) / dt;
		dupz_r = (r_upz2[i * Zn + j] - r_upz1[i * Zn + j]) / dt;
		dusx_r = (r_usx2[i * Zn + j] - r_usx1[i * Zn + j]) / dt;
		dusz_r = (r_usz2[i * Zn + j] - r_usz1[i * Zn + j]) / dt;
	
		Ep_r.x = -r_theta[i * Zn + j] * dupx_r;
		Ep_r.y = -r_theta[i * Zn + j] * dupz_r;

		Es_r.x = r_omega[i * Zn + j] * dusz_r;
		Es_r.y = -r_omega[i * Zn + j] * dusx_r;

				 
		if (Ep_s.y >= 0)
		{
			upx_up[i * Zn + j] = 0.0;
			upx_down[i * Zn + j] = upx0[i * Zn + j];
			upz_up[i * Zn + j] = 0.0;
			upz_down[i * Zn + j] = upz0[i * Zn + j];
		}
		else
		{
			upx_up[i * Zn + j] = upx0[i * Zn + j];
			upx_down[i * Zn + j] = 0.0;
			upz_up[i * Zn + j] = upz0[i * Zn + j];
			upz_down[i * Zn + j] = 0.0;
		}


		if (Ep_s.x >= 0)
		{
			upx_left[i * Zn + j] = 0.0;
			upx_right[i * Zn + j] = upx0[i * Zn + j];
			upz_left[i * Zn + j] = 0.0;
			upz_right[i * Zn + j] = upz0[i * Zn + j];
		}
		else
		{
			upx_left[i * Zn + j] = upx0[i * Zn + j];
			upx_right[i * Zn + j] = 0.0;
			upz_left[i * Zn + j] = upz0[i * Zn + j];
			upz_right[i * Zn + j] = 0.0;
		}


		if (Ep_r.y >= 0)
		{
			r_upx_up[i * Zn + j] = 0.0;
			r_upx_down[i * Zn + j] = r_upx2[i * Zn + j];
			r_upz_up[i * Zn + j] = 0.0;
			r_upz_down[i * Zn + j] = r_upz2[i * Zn + j];
		}
		else
		{
			r_upx_up[i * Zn + j] = r_upx2[i * Zn + j];
			r_upx_down[i * Zn + j] = 0.0;
			r_upz_up[i * Zn + j] = r_upz2[i * Zn + j];
			r_upz_down[i * Zn + j] = 0.0;
		}

		if (Ep_r.x >= 0)
		{
			r_upx_left[i * Zn + j] = 0.0;
			r_upx_right[i * Zn + j] = r_upx2[i * Zn + j];
			r_upz_left[i * Zn + j] = 0.0;
			r_upz_right[i * Zn + j] = r_upz2[i * Zn + j];
		}
		else
		{
			r_upx_left[i * Zn + j] = r_upx2[i * Zn + j];
			r_upx_right[i * Zn + j] = 0.0;
			r_upz_left[i * Zn + j] = r_upz2[i * Zn + j];
			r_upz_right[i * Zn + j] = 0.0;
		}



	
		if (Es_r.y >= 0)
		{
			r_usx_up[i * Zn + j] = 0.0;
			r_usx_down[i * Zn + j] = r_usx2[i * Zn + j];
			r_usz_up[i * Zn + j] = 0.0;
			r_usz_down[i * Zn + j] = r_usz2[i * Zn + j];
		}
		else
		{
			r_usx_up[i * Zn + j] = r_usx2[i * Zn + j];
			r_usx_down[i * Zn + j] = 0.0;
			r_usz_up[i * Zn + j] = r_usz2[i * Zn + j];
			r_usz_down[i * Zn + j] = 0.0;
		}

		if (Es_r.x >= 0)
		{
			r_usx_left[i * Zn + j] = 0.0;
			r_usx_right[i * Zn + j] = r_usx2[i * Zn + j];
			r_usz_left[i * Zn + j] = 0.0;
			r_usz_right[i * Zn + j] = r_usz2[i * Zn + j];
		}

		else
		{
			r_usx_left[i * Zn + j] = r_usx2[i * Zn + j];
			r_usx_right[i * Zn + j] = 0.0;
			r_usz_left[i * Zn + j] = r_usz2[i * Zn + j];
			r_usz_right[i * Zn + j] = 0.0;
		}

	}

}

void main_RTM(int myid, struct Parameter& para, int Xn, int  Zn, int  L, int Tn, int sxnum, int shotnum, int sy, int gy,
	float dx, float dz, float dt, float FM, float* vp0, float* vs0, float* record_ux, float* record_uz, float* illumination, float* migration_pp, float* migration_ps, int WriteSnapshot)
{
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((Zn + dimBlock.x - 1) / dimBlock.x, (Xn + dimBlock.y - 1) / dimBlock.y, 1);
	int blockx = 512;  
	FILE* fp;
	char filename[1024];
	int Xn1 = Xn - 2 * L;
	int Zn1 = Zn - 2 * L;


	cudaMemcpy(para.vp0, vp0, Zn * Xn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(para.vs0, vs0, Zn * Xn * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemcpy(para.record_ux, record_ux, Tn * Xn1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(para.record_uz, record_uz, Tn * Xn1 * sizeof(float), cudaMemcpyHostToDevice);



	cudaMemset(para.upx0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upx1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upx2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.usx0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.usx1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.usx2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.usz0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.usz1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.usz2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.ux, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.uz, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.theta, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.omega, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.duzdx, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.duzdz, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.duxdx, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.duxdz, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.r_upx0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upx1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upx2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usx0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usx1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usx2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_ux, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_uz, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_theta, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_omega, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_duzdx, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_duzdz, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_duxdx, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_duxdz, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.upx_down, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upx_up, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upx_left, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upx_right, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz_down, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz_up, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz_left, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.upz_right, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.r_upx_down, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upx_up, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upx_left, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upx_right, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz_down, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz_up, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz_left, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_upz_right, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.r_usx_down, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usx_up, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usx_left, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usx_right, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz_down, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz_up, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz_left, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.r_usz_right, 0, Zn * Xn * sizeof(float));


	cudaMemset(para.damp, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.ux_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.uz_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.theta_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.omega_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.duzdx_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.duzdz_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.duxdz_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));
	cudaMemset(para.duxdx_bdr, 0, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float));

	cudaMemset(para.illumination_pp, 0, Zn1 * Xn1 * sizeof(float));
	cudaMemset(para.migration_pp, 0, Zn1 * Xn1 * sizeof(float));
	cudaMemset(para.migration_ps, 0, Zn1 * Xn1 * sizeof(float));

	cudaMemset(para.migration_poynting_pp, 0, Zn1* Xn1 * sizeof(float));
	cudaMemset(para.migration_poynting_ps, 0, Zn1* Xn1 * sizeof(float));

	get_d0 << <1, 1 >> > (L, dx);
	pml_coef << <dimGrid, dimBlock >> > (Xn, Zn, L, para.vp0, para.damp);

	
	int flag = 1;
	for (int it = 0; it < Tn; it++)
	{

		add_source << <1, 1 >> > (Zn, sxnum, sy, it, dt, FM, para.theta);

		update_u << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.upx0, para.upx1, para.upx2,
			para.upz0, para.upz1, para.upz2,
			para.usx0, para.usx1, para.usx2,
			para.usz0, para.usz1, para.usz2,
			para.ux, para.uz,
			para.theta, para.omega,
			para.duzdx, para.duzdz, para.duxdz, para.duxdx,
			flag);

		update_stress << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.ux, para.uz,
			para.theta, para.omega,
			para.duzdx, para.duzdz, para.duxdz, para.duxdx,
			flag);


		if (sxnum == 600 && it > 0 && it % 400 == 0)
		{
			cudaMemcpy(para.h_temp, para.upx2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/snapshot/upx2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.upz2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/snapshot/upz2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.usx2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/snapshot/usx2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.usz2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/snapshot/usz2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

		}

		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.ux_bdr, para.ux, true);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.uz_bdr, para.uz, true);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.theta_bdr, para.theta, true);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.omega_bdr, para.omega, true);

		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duzdx_bdr, para.duzdx, true);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duzdz_bdr, para.duzdz, true);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duxdz_bdr, para.duxdz, true);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duxdx_bdr, para.duxdx, true);

		cal_illumination << <dimGrid, dimBlock >> > (Xn, Zn, L, para.upx2, para.upz2, para.illumination_pp, it, Tn);

		
		if (it != Tn - 1) 
		{
			swap_pointer(para.upx0, para.upx1);
			swap_pointer(para.upx1, para.upx2);
			swap_pointer(para.upz0, para.upz1);
			swap_pointer(para.upz1, para.upz2);

			swap_pointer(para.usx0, para.usx1);
			swap_pointer(para.usx1, para.usx2);
			swap_pointer(para.usz0, para.usz1);
			swap_pointer(para.usz1, para.usz2);

		}

	}

	
	cudaMemset(para.theta, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.omega, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.duzdx, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.duzdz, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.duxdx, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.duxdz, 0, Zn* Xn * sizeof(float));

	cudaMemset(para.upx0, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.upz0, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.usx0, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.usz0, 0, Zn* Xn * sizeof(float));


	
	for (int it = Tn - 1; it >= 0; it--)
	{
		
		flag = 2;

		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.ux_bdr, para.ux, false);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.uz_bdr, para.uz, false);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.theta_bdr, para.theta, false);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.omega_bdr, para.omega, false);

		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duzdx_bdr, para.duzdx, false);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duzdz_bdr, para.duzdz, false);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duxdz_bdr, para.duxdz, false);
		wavefield_bdr << <(int)((2 * N * Xn1 + 2 * N * Zn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, L, it, para.duxdx_bdr, para.duxdx, false);

		update_stress << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.ux, para.uz,
			para.theta, para.omega,
			para.duzdx, para.duzdz, para.duxdz, para.duxdx,
			flag);

		update_u << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.upx0, para.upx1, para.upx2,
			para.upz0, para.upz1, para.upz2,
			para.usx0, para.usx1, para.usx2,
			para.usz0, para.usz1, para.usz2,
			para.ux, para.uz,
			para.theta, para.omega,
			para.duzdx, para.duzdz, para.duxdz, para.duxdx,
			flag);





	

		if (sxnum == 600 && it > 0 && it % 400 == 0)
		{
			cudaMemcpy(para.h_temp, para.upx0, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/reconstruct/upx0_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.upz0, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/reconstruct/upz0_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.usx0, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/reconstruct/usx0_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.usz0, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/reconstruct/usz0_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.ux, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/reconstruct/ux_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.uz, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/reconstruct/uz_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);


		}

		
		swap_pointer(para.upx2, para.upx1);
		swap_pointer(para.upx1, para.upx0);
		swap_pointer(para.upz2, para.upz1);
		swap_pointer(para.upz1, para.upz0);
		swap_pointer(para.usx2, para.usx1);
		swap_pointer(para.usx1, para.usx0);
		swap_pointer(para.usz2, para.usz1);
		swap_pointer(para.usz1, para.usz0);


		
		flag = 3;

		shot_record << <(int)((Xn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, Tn, it, L, gy, para.r_ux, para.record_ux, false);
		shot_record << <(int)((Xn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, Tn, it, L, gy, para.r_uz, para.record_uz, false);

		update_stress << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.r_ux, para.r_uz,
			para.r_theta, para.r_omega,
			para.r_duzdx, para.r_duzdz, para.r_duxdz, para.r_duxdx,
			flag);

		update_u << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.r_upx0, para.r_upx1, para.r_upx2,
			para.r_upz0, para.r_upz1, para.r_upz2,
			para.r_usx0, para.r_usx1, para.r_usx2,
			para.r_usz0, para.r_usz1, para.r_usz2,
			para.r_ux, para.r_uz,
			para.r_theta, para.r_omega,
			para.r_duzdx, para.r_duzdz, para.r_duxdz, para.r_duxdx,
			flag);

		
		if (sxnum == 600 && it > 0 && it % 400 == 0)
		{
			cudaMemcpy(para.h_temp, para.r_upx2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/backward/upx2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.r_upz2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/backward/upz2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.r_usx2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/backward/usx2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.r_usz2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/backward/usz2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

		}

		
		swap_pointer(para.r_upx0, para.r_upx1);
		swap_pointer(para.r_upx1, para.r_upx2);
		swap_pointer(para.r_upz0, para.r_upz1);
		swap_pointer(para.r_upz1, para.r_upz2);
		swap_pointer(para.r_usx0, para.r_usx1);
		swap_pointer(para.r_usx1, para.r_usx2);
		swap_pointer(para.r_usz0, para.r_usz1);
		swap_pointer(para.r_usz1, para.r_usz2);

		cal_migration << <dimGrid, dimBlock >> > (Xn, Zn, L, para.upx0, para.upz0, para.r_upx2, para.r_upz2, para.migration_pp);
		cal_migration << <dimGrid, dimBlock >> > (Xn, Zn, L, para.upx0, para.upz0, para.r_usx2, para.r_usz2, para.migration_ps);

		poynting << <dimGrid, dimBlock >> > (Xn, Zn, L, para.r_upx2, para.r_upz2, para.r_usx2, para.r_usz2, para.r_upx1, para.r_upz1, para.r_usx1, para.r_usz1,
			para.upx0, para.upz0, para.upx1, para.upz1, para.theta,
			para.r_theta, para.r_omega,
			para.r_upx_up, para.r_upx_down, para.r_upx_left, para.r_upx_right, para.r_upz_up, para.r_upz_down, para.r_upz_left, para.r_upz_right,
			para.r_usx_up, para.r_usx_down, para.r_usx_left, para.r_usx_right, para.r_usz_up, para.r_usx_down, para.r_usz_left, para.r_usx_right,
			para.upx_up, para.upx_down, para.upx_left, para.upx_right, para.upz_up, para.upz_down, para.upz_left, para.upz_right, para.r_duxdz, para.r_duzdz, para.r_duxdx, para.r_duzdx, dt);
		
		cal_migration << <dimGrid, dimBlock >> > (Xn, Zn, L, para.upx_down, para.upz_down, para.upx_up, para.upz_up,para.r_upx_up, para.r_upz_up, para.r_upx_down, para.r_upz_down,para.r_usx_up, para.r_usz_up, para.r_usx_down, para.r_usz_down,
			para.upx_right, para.upz_right, para.r_upx_left, para.r_upz_left, para.r_usx_left, para.r_usz_left, para.upx_left, para.upz_left, para.r_upx_right, para.r_upz_right, para.r_usx_right, para.r_usz_right,
			para.migration_poynting_pp,
			para.migration_poynting_ps);
	
	}

	migration_illum << <dimGrid, dimBlock >> > (Xn, Zn, L, para.illumination_pp, para.migration_pp);
	migration_illum << <dimGrid, dimBlock >> > (Xn, Zn, L, para.illumination_pp, para.migration_ps);

	cudaMemcpy(migration_pp, para.migration_pp, Xn1 * Zn1 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(migration_ps, para.migration_ps, Xn1* Zn1 * sizeof(float), cudaMemcpyDeviceToHost);

}






void cuda_Device_malloc(int myid, struct Parameter& para, int Xn, int Xn1, int Zn, int Zn1, int Tn)
{

	cudaError_t ct;

	para.h_temp = (float*)malloc(Zn * Xn * sizeof(float));

	ct = cudaMalloc(&para.vp0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.vs0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);


	ct = cudaMalloc(&para.upx0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upx1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upx2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.usx0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.usx1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.usx2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.usz0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.usz1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.usz2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.ux, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.uz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.theta, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.omega, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duzdx, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duzdz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duxdx, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duxdz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.r_upx0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upx1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upx2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usx0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usx1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usx2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_ux, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_uz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_theta, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_omega, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_duzdx, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_duzdz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_duxdx, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_duxdz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.upx_down, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upx_up, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upx_left, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upx_right, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz_down, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz_up, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz_left, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.upz_right, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.r_upx_down, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upx_up, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upx_left, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upx_right, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz_down, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz_up, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz_left, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_upz_right, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.r_usx_down, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usx_up, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usx_left, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usx_right, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz_down, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz_up, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz_left, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.r_usz_right, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.record_ux, Tn * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.record_uz, Tn * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.damp, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.ux_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.uz_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.theta_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.omega_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duzdx_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duzdz_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duxdz_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.duxdx_bdr, Tn * (2 * N * Xn1 + 2 * N * Zn1) * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.illumination_pp, Zn1 * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.migration_pp, Zn1 * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.migration_ps, Zn1 * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.migration_poynting_pp, Zn1 * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.migration_poynting_ps, Zn1 * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);


	memset(para.h_temp, 0, Zn * Xn * sizeof(float));

}
void cuda_Device_free(struct Parameter& para)
{
	free(para.h_temp);

	cudaFree(para.vp0);
	cudaFree(para.vs0);

	cudaFree(para.upx0);
	cudaFree(para.upx1);
	cudaFree(para.upx2);
	cudaFree(para.upz0);
	cudaFree(para.upz1);
	cudaFree(para.upz2);
	cudaFree(para.usx0);
	cudaFree(para.usx1);
	cudaFree(para.usx2);
	cudaFree(para.usz0);
	cudaFree(para.usz1);
	cudaFree(para.usz2);
	cudaFree(para.ux);
	cudaFree(para.uz);
	cudaFree(para.theta);
	cudaFree(para.omega);
	cudaFree(para.duzdx);
	cudaFree(para.duzdz);
	cudaFree(para.duxdx);
	cudaFree(para.duxdz);

	cudaFree(para.r_upx0);
	cudaFree(para.r_upx1);
	cudaFree(para.r_upx2);
	cudaFree(para.r_upz0);
	cudaFree(para.r_upz1);
	cudaFree(para.r_upz2);
	cudaFree(para.r_usx0);
	cudaFree(para.r_usx1);
	cudaFree(para.r_usx2);
	cudaFree(para.r_usz0);
	cudaFree(para.r_usz1);
	cudaFree(para.r_usz2);
	cudaFree(para.r_ux);
	cudaFree(para.r_uz);
	cudaFree(para.r_theta);
	cudaFree(para.r_omega);
	cudaFree(para.r_duzdx);
	cudaFree(para.r_duzdz);
	cudaFree(para.r_duxdx);
	cudaFree(para.r_duxdz);

	cudaFree(para.upx_down);
	cudaFree(para.upx_up);
	cudaFree(para.upx_left);
	cudaFree(para.upx_right);
	cudaFree(para.upz_down);
	cudaFree(para.upz_up);
	cudaFree(para.upz_left);
	cudaFree(para.upz_right);

	cudaFree(para.r_upx_down);
	cudaFree(para.r_upx_up);
	cudaFree(para.r_upx_left);
	cudaFree(para.r_upx_right);
	cudaFree(para.r_upz_down);
	cudaFree(para.r_upz_up);
	cudaFree(para.r_upz_left);
	cudaFree(para.r_upz_right);

	cudaFree(para.r_usx_down);
	cudaFree(para.r_usx_up);
	cudaFree(para.r_usx_left);
	cudaFree(para.r_usx_right);
	cudaFree(para.r_usz_down);
	cudaFree(para.r_usz_up);
	cudaFree(para.r_usz_left);
	cudaFree(para.r_usz_right);

	cudaFree(para.record_ux);
	cudaFree(para.record_uz);
	cudaFree(para.damp);

	cudaFree(para.ux_bdr);
	cudaFree(para.uz_bdr);
	cudaFree(para.theta_bdr);
	cudaFree(para.omega_bdr);
	cudaFree(para.duzdx_bdr);
	cudaFree(para.duzdz_bdr);
	cudaFree(para.duxdx_bdr);
	cudaFree(para.duxdz_bdr);

	cudaFree(para.illumination_pp);
	
	cudaFree(para.migration_pp);
	cudaFree(para.migration_ps);
	cudaFree(para.migration_poynting_pp);
	cudaFree(para.migration_poynting_ps);
}
