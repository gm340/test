#define _CRT_SECURE_NO_WARNINGS
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "functions.h"

#define PI 3.141592653
#define N 6           // space 2N order difference
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__));

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(err), cudaGetErrorString(err), file, line);
		exit(int(err));
	}
}

__device__ float c2[N + 1] = { -2.9828 / 2,1.7143,-0.2679,0.0529,-0.0089,0.0010,-0.0001 }; // second-order derivative      c2[0]/2 to simplify calculate derivative
__device__ float c1[N + 1] = { 0.0000,0.8571,-0.2679,0.0794,-0.0179,0.0026,-0.0002 };       // first-order derivative
__device__ float d0;  // d0 for PML coefficient

__device__ float a[6] = { 1.2213365, -9.6931458e-2, 1.7447662e-2, -2.9672895e-3, 3.5900540e-4, -2.1847812e-5 };

// creat 1D float array and initialize
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

// sawp pointer
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

// read model parameter
void readFile(char FNvelocity[], char FNvs0[], float* vp0, float* vs0, int Xn, int Zn, int L)
{
	int i, j, idx;
	float vmax, vmin;
	float emax, emin;
	FILE* fp1, * fp2;
	// check file
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

			// get min and max value
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

// initialize boundary
void ini_bdr(int Xn, int Zn, int L, float* ee)
{
	int ix, iz, idx;
	for (idx = 0; idx < Xn * Zn; idx++)
	{
		ix = idx / Zn;
		iz = idx % Zn;

		// left & right
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

		// up and bottom
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

// get d0 for PML coefficient
__global__ void get_d0(int L, float dx)
{
	d0 = 1.5 * log(1e9) / (L * dx);
}

// get PML coefficient
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
		if (ix < L) // left
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(L - ix) / L, 2.0);
		}
		else if (ix >= Xn - L)  // right
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(ix - (Xn - L - 1)) / L, 2.0);
		}
		else if (ix >= L && ix < Xn - L && iz < L)// up
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(L - iz) / L, 2.0);
		}
		else if (ix >= L && ix < Xn - L && iz >= Zn - L) // bottom
		{
			damp[idx] = d0 * vp0[idx] * pow((double)(iz - (Zn - L - 1)) / L, 2.0);
		}
		else // center
		{
			damp[idx] = 0.0;
		}
	}

}

// add source
__global__ void add_source(
	int Zn,
	int shotx, int shotz,
	int it,
	float dt,
	float FM,
	float* u
)
{
	// one thread
	// Ricker wavelet
	int tdelay = ceil(1.0 / (FM * dt));
	float wavelet = (1.0 - 2.0 * PI * PI * FM * FM * dt * dt * (it - tdelay) * (it - tdelay)) * exp(-PI * PI * FM * FM * dt * dt * (it - tdelay) * (it - tdelay));
	u[shotx * Zn + shotz] += 1.0 * wavelet;
}

// calculation u_z
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
		// first-order derivative u_z
		for (m = 0; m <= N; m++)
		{
			u_z[idx] += c1[m] * (u[ix * Zn + iz + m] - u[ix * Zn + iz - m]) / dz;
		}
	}

}

// calculate u_xz
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
	float* ux0, float* ux1, float* ux2,
	float* uz0, float* uz1, float* uz2,
	float* taoxx, float* taozz, float* taoxz,
	int flag)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i * Zn + j;

	int m;

	float taoxx_x = 0.0f;
	float taozz_z = 0.0f;
	float taoxz_x = 0.0f;
	float taoxz_z = 0.0f;

	if (flag == 1) // forward
	{
		if (i >= N && i < Xn - N && j >= N && j < Zn - N)
		{

			taoxx_x = (a[0] * (taoxx[(i + 1) * Zn + j] - taoxx[(i - 0) * Zn + j])
				+ a[1] * (taoxx[(i + 2) * Zn + j] - taoxx[(i - 1) * Zn + j])
				+ a[2] * (taoxx[(i + 3) * Zn + j] - taoxx[(i - 2) * Zn + j])
				+ a[3] * (taoxx[(i + 4) * Zn + j] - taoxx[(i - 3) * Zn + j])
				+ a[4] * (taoxx[(i + 5) * Zn + j] - taoxx[(i - 4) * Zn + j])
				+ a[5] * (taoxx[(i + 6) * Zn + j] - taoxx[(i - 5) * Zn + j])) / dx;

			taozz_z = (a[0] * (taozz[(i)*Zn + j + 1] - taozz[(i)*Zn + j - 0])
				+ a[1] * (taozz[(i)*Zn + j + 2] - taozz[(i)*Zn + j - 1])
				+ a[2] * (taozz[(i)*Zn + j + 3] - taozz[(i)*Zn + j - 2])
				+ a[3] * (taozz[(i)*Zn + j + 4] - taozz[(i)*Zn + j - 3])
				+ a[4] * (taozz[(i)*Zn + j + 5] - taozz[(i)*Zn + j - 4])
				+ a[5] * (taozz[(i)*Zn + j + 6] - taozz[(i)*Zn + j - 5])) / dz;

			taoxz_x = (a[0] * (taoxz[(i + 0) * Zn + j] - taoxz[(i - 1) * Zn + j])
				+ a[1] * (taoxz[(i + 1) * Zn + j] - taoxz[(i - 2) * Zn + j])
				+ a[2] * (taoxz[(i + 2) * Zn + j] - taoxz[(i - 3) * Zn + j])
				+ a[3] * (taoxz[(i + 3) * Zn + j] - taoxz[(i - 4) * Zn + j])
				+ a[4] * (taoxz[(i + 4) * Zn + j] - taoxz[(i - 5) * Zn + j])
				+ a[5] * (taoxz[(i + 5) * Zn + j] - taoxz[(i - 6) * Zn + j])) / dx;

			taoxz_z = (a[0] * (taoxz[(i)*Zn + j + 0] - taoxz[(i)*Zn + j - 1])
				+ a[1] * (taoxz[(i)*Zn + j + 1] - taoxz[(i)*Zn + j - 2])
				+ a[2] * (taoxz[(i)*Zn + j + 2] - taoxz[(i)*Zn + j - 3])
				+ a[3] * (taoxz[(i)*Zn + j + 3] - taoxz[(i)*Zn + j - 4])
				+ a[4] * (taoxz[(i)*Zn + j + 4] - taoxz[(i)*Zn + j - 5])
				+ a[5] * (taoxz[(i)*Zn + j + 5] - taoxz[(i)*Zn + j - 6])) / dz;


			ux2[idx] = (2.0 - damp[idx] * dt) * ux1[idx] - (1.0 - damp[idx] * dt) * ux0[idx] + dt * dt * (taoxx_x + taoxz_z);
			uz2[idx] = (2.0 - damp[idx] * dt) * uz1[idx] - (1.0 - damp[idx] * dt) * uz0[idx] + dt * dt * (taozz_z + taoxz_x);



		}
	}


}

__global__ void update_stress(
	int Xn, int Zn, int L,
	float dx, float dz, float dt,
	float* vp0, float* vs0,
	float* damp,
	float* ux, float* uz,
	float* taoxx, float* taozz, float* taoxz,	
	int flag
)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = i * Zn + j;

	int m;

	float dvxdx = 0.0f;
	float dvzdz = 0.0f;
	float dvxdz = 0.0f;
	float dvzdx = 0.0f;


	if (flag == 1) // forward
	{
		if (i >= N && i < Xn - N && j >= N && j < Zn - N)
		{

			dvxdx = (a[0] * (ux[(i + 0) * Zn + j] - ux[(i - 1) * Zn + j])
				+ a[1] * (ux[(i + 1) * Zn + j] - ux[(i - 2) * Zn + j])
				+ a[2] * (ux[(i + 2) * Zn + j] - ux[(i - 3) * Zn + j])
				+ a[3] * (ux[(i + 3) * Zn + j] - ux[(i - 4) * Zn + j])
				+ a[4] * (ux[(i + 4) * Zn + j] - ux[(i - 5) * Zn + j])
				+ a[5] * (ux[(i + 5) * Zn + j] - ux[(i - 6) * Zn + j])) / dx;

			dvxdz = (a[0] * (ux[(i)*Zn + j + 1] - ux[(i)*Zn + j - 0])
				+ a[1] * (ux[(i)*Zn + j + 2] - ux[(i)*Zn + j - 1])
				+ a[2] * (ux[(i)*Zn + j + 3] - ux[(i)*Zn + j - 2])
				+ a[3] * (ux[(i)*Zn + j + 4] - ux[(i)*Zn + j - 3])
				+ a[4] * (ux[(i)*Zn + j + 5] - ux[(i)*Zn + j - 4])
				+ a[5] * (ux[(i)*Zn + j + 6] - ux[(i)*Zn + j - 5])) / dz;

			dvzdz = (a[0] * (uz[(i)*Zn + j + 0] - uz[(i)*Zn + j - 1])
				+ a[1] * (uz[(i)*Zn + j + 1] - uz[(i)*Zn + j - 2])
				+ a[2] * (uz[(i)*Zn + j + 2] - uz[(i)*Zn + j - 3])
				+ a[3] * (uz[(i)*Zn + j + 3] - uz[(i)*Zn + j - 4])
				+ a[4] * (uz[(i)*Zn + j + 4] - uz[(i)*Zn + j - 5])
				+ a[5] * (uz[(i)*Zn + j + 5] - uz[(i)*Zn + j - 6])) / dz;

			dvzdx = (a[0] * (uz[(i + 1) * Zn + j] - uz[(i - 0) * Zn + j])
				+ a[1] * (uz[(i + 2) * Zn + j] - uz[(i - 1) * Zn + j])
				+ a[2] * (uz[(i + 3) * Zn + j] - uz[(i - 2) * Zn + j])
				+ a[3] * (uz[(i + 4) * Zn + j] - uz[(i - 3) * Zn + j])
				+ a[4] * (uz[(i + 5) * Zn + j] - uz[(i - 4) * Zn + j])
				+ a[5] * (uz[(i + 6) * Zn + j] - uz[(i - 5) * Zn + j])) / dx;


			taoxx[idx] = vp0[idx] * vp0[idx] * dvxdx + (vp0[idx] * vp0[idx] - 2 * vs0[idx] * vs0[idx]) * dvzdz;
			taozz[idx] = (vp0[idx] * vp0[idx] - 2 * vs0[idx] * vs0[idx]) * dvxdx + vp0[idx] * vp0[idx] * dvzdz;
			taoxz[idx] = (vs0[idx] * vs0[idx]) * dvxdz + (vs0[idx] * vs0[idx]) * dvzdx;

		

		}
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
		// record u
		if (symbol)
		{
			record[ix * Tn + it] = u[(ix + L) * Zn + recdep];
		}
		// load receiver u
		else
		{
			u[(ix + L) * Zn + recdep] = record[ix * Tn + it];
		}
	}
}

__global__ void mute_directwave(int Xn, int Zn,
	int L, int Tn,
	int shotx, int shotz, int recdep,
	float FM, float dt,
	float dx, float dz,
	float* vp0, float* epsilon,
	float* record,
	int tt
)
{
	int it = blockIdx.x * blockDim.x + threadIdx.x;
	int ix = blockIdx.y * blockDim.y + threadIdx.y;
	float xx, zz;
	float t0, t1;

	if (ix < Xn - 2 * L && it < Tn)
	{
		xx = abs(ix + L - shotx) * dx;
		zz = abs(shotz - recdep) * dz;
		t0 = sqrt(xx * xx + zz * zz) / (vp0[(ix + L) * Zn + recdep] * sqrt(1 + 2 * epsilon[(ix + L) * Zn + recdep]));
		t1 = t0 + 2 / FM;
		if (it > (int)(t0 / dt) - tt && it < (int)(t1 / dt) + tt)
		{
			record[ix * Tn + it] = 0.0;
		}
	}
}


void main_Forward(int myid, struct Parameter& para, int Xn, int  Zn, int  L, int Tn, int sxnum, int shotnum, int sy, int gy,
	float dx, float dz, float dt, float FM, float* vp0, float* vs0, float* record_ux, float* record_ux_mute, float* record_uz, float* record_uz_mute, int WriteSnapshot)
{

	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid((Zn + dimBlock.x - 1) / dimBlock.x, (Xn + dimBlock.y - 1) / dimBlock.y, 1);
	int blockx = 512;  // 1-D block
	FILE* fp;
	char filename[1024];
	int Xn1 = Xn - 2 * L;
	int Zn1 = Zn - 2 * L;

	float* vp0_hom = initializeArray(Zn, Xn, 0.0);
	float* vs0_hom = initializeArray(Zn, Xn, 0.0);

	////////////////			forward use model		////////////////
	cudaMemcpy(para.vp0, vp0, Zn * Xn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(para.vs0, vs0, Zn * Xn * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemset(para.ux0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.ux1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.ux2, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.uz0, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.uz1, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.uz2, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.taoxx, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.taozz, 0, Zn * Xn * sizeof(float));
	cudaMemset(para.taoxz, 0, Zn * Xn * sizeof(float));


	cudaMemset(para.damp, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.record_ux, 0, Tn * Xn1 * sizeof(float));
	cudaMemset(para.record_uz, 0, Tn * Xn1 * sizeof(float));

	
	get_d0 << <1, 1 >> > (L, dx);
	pml_coef << <dimGrid, dimBlock >> > (Xn, Zn, L, para.vp0, para.damp);

	int flag = 1;
	for (int it = 0; it < Tn; it++)
	{
		if (myid == 1 && it % 1000 == 0)
		{
			//printf("Forward it = %d on Processor %d \n", it, myid);
		}


		add_source << <1, 1 >> > (Zn, sxnum, sy, it, dt, FM, para.taoxx);
		add_source << <1, 1 >> > (Zn, sxnum, sy, it, dt, FM, para.taozz);


		update_u << <dimGrid, dimBlock >> > (
			Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.ux0, para.ux1, para.ux2,
			para.uz0, para.uz1, para.uz2,
			para.taoxx, para.taozz, para.taoxz,
			flag);

		update_stress << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.ux2, para.uz2,
			para.taoxx, para.taozz, para.taoxz,			
			flag);




		if ( sxnum== 800&& it > 0 && it % 400 == 0)
		{
			cudaMemcpy(para.h_temp, para.ux2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/displacement_stress/snapshot/ux2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.uz2, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/displacement_stress/snapshot/uz2_%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.taoxx, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/displacement_stress/snapshot/taoxx%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.taozz, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/displacement_stress/snapshot/taozz%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);

			cudaMemcpy(para.h_temp, para.taoxz, Zn * Xn * sizeof(float), cudaMemcpyDeviceToHost);
			sprintf(filename, "/public/home/scw/Desktop/2023413/displacement_stress/snapshot/taoxz%d_%d_%d.bin", it, Xn1, Zn1);
			fp = fopen(filename, "wb");
			for (int i = L; i < Xn - L; i++) {
				for (int j = L; j < Zn - L; j++) {
					fwrite(&para.h_temp[i * Zn + j], sizeof(float), 1, fp);
				}
			}
			fclose(fp);
		}

		shot_record << <(int)((Xn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, Tn, it, L, gy, para.ux2, para.record_ux, true);
		shot_record << <(int)((Xn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, Tn, it, L, gy, para.uz2, para.record_uz, true);

		// swap u0, u1, u2 to update wavefield
		if (it != Tn - 1) // dont swap for latest time
		{
			swap_pointer(para.ux0, para.ux1);
			swap_pointer(para.ux1, para.ux2);
			swap_pointer(para.uz0, para.uz1);
			swap_pointer(para.uz1, para.uz2);


		}

	} // it


	////////////////			forward use homogeneous model		////////////////
	// 
	// set homogeneous model
	for (int i = 0; i < Xn * Zn; i++)
	{
		vp0_hom[i] = vp0[sxnum * Zn + sy];
		vs0_hom[i] = vs0[sxnum * Zn + sy];
	}

	cudaMemcpy(para.vp0, vp0_hom, Zn * Xn * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(para.vs0, vs0_hom, Zn * Xn * sizeof(float), cudaMemcpyHostToDevice);


	cudaMemset(para.ux0, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.ux1, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.ux2, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.uz0, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.uz1, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.uz2, 0, Zn* Xn * sizeof(float));

	cudaMemset(para.taoxx, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.taozz, 0, Zn* Xn * sizeof(float));
	cudaMemset(para.taoxz, 0, Zn* Xn * sizeof(float));

	cudaMemset(para.damp, 0, Zn * Xn * sizeof(float));

	cudaMemset(para.record_ux_mute, 0, Tn * Xn1 * sizeof(float));
	cudaMemset(para.record_uz_mute, 0, Tn * Xn1 * sizeof(float));

	// get d0 and PML absorb coefficient
	get_d0 << <1, 1 >> > (L, dx);
	pml_coef << <dimGrid, dimBlock >> > (Xn, Zn, L, para.vp0, para.damp);

	flag = 1;
	for (int it = 0; it < Tn; it++)
	{

		add_source << <1, 1 >> > (Zn, sxnum, sy, it, dt, FM, para.taoxx);
		add_source << <1, 1 >> > (Zn, sxnum, sy, it, dt, FM, para.taozz);


		update_u << <dimGrid, dimBlock >> > (
			Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.ux0, para.ux1, para.ux2,
			para.uz0, para.uz1, para.uz2,
			para.taoxx, para.taozz, para.taoxz,
			flag);

		update_stress << <dimGrid, dimBlock >> > (Xn, Zn, L,
			dx, dz, dt,
			para.vp0, para.vs0,
			para.damp,
			para.ux2, para.uz2,
			para.taoxx, para.taozz, para.taoxz,			
			flag);

		shot_record << <(int)((Xn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, Tn, it, L, gy, para.ux2, para.record_ux_mute, true);
		shot_record << <(int)((Xn1 + blockx - 1) / blockx), blockx >> > (Xn, Zn, Tn, it, L, gy, para.uz2, para.record_uz_mute, true);

		// swap u0, u1, u2 to update wavefield
		if (it != Tn - 1) // dont swap for latest time
		{
			swap_pointer(para.ux0, para.ux1);
			swap_pointer(para.ux1, para.ux2);
			swap_pointer(para.uz0, para.uz1);
			swap_pointer(para.uz1, para.uz2);


		}

	} // it

	// cpy record & record_mute
	cudaMemcpy(record_ux, para.record_ux, Xn1* Tn * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(record_ux_mute, para.record_ux_mute, Xn1* Tn * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(record_uz, para.record_uz, Xn1 * Tn * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(record_uz_mute, para.record_uz_mute, Xn1 * Tn * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < Xn1; i++)
		for (int j = 0; j < Tn; j++)
		{
			int idx = i * Tn + j;
			record_ux_mute[idx] = record_ux[idx] - record_ux_mute[idx];
			record_uz_mute[idx] = record_uz[idx] - record_uz_mute[idx];
		}

	//// mute direct wave
	//cudaMemcpy(para.record_mute, para.record, Xn1* Tn * sizeof(float), cudaMemcpyDeviceToDevice);
	//dim3 grid1((Tn + dimBlock.x - 1) / dimBlock.x, (Xn1 + dimBlock.y - 1) / dimBlock.y, 1);
	//mute_directwave << < grid1, dimBlock >> > (Xn, Zn, L, Tn, sxnum, sy, gy, FM, dt, dx, dz, para.vp0, para.epsilon, para.record_mute, 30);

	free(vp0_hom);
	free(vs0_hom);

}



void cuda_Device_malloc(int myid, struct Parameter &para,  int Xn, int Xn1, int Zn, int Zn1, int Tn)
{

	cudaError_t ct;

	para.h_temp = (float*)malloc(Zn * Xn * sizeof(float));

	ct = cudaMalloc(&para.vp0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.vs0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.ux0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.ux1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.ux2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.uz0, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.uz1, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.uz2, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.taoxx, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.taozz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.taoxz, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.damp, Zn * Xn * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

	ct = cudaMalloc(&para.record_ux, Tn * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.record_ux_mute, Tn * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.record_uz, Tn * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);
	ct = cudaMalloc(&para.record_uz_mute, Tn * Xn1 * sizeof(float)); HandleError(ct, "cudaMallocDevice", __LINE__);

}
void cuda_Device_free(struct Parameter& para)
{
	free(para.h_temp);

	cudaFree(para.vp0);
	cudaFree(para.vs0);

	cudaFree(para.ux0);
	cudaFree(para.ux1);
	cudaFree(para.ux2);
	cudaFree(para.uz0);
	cudaFree(para.uz1);
	cudaFree(para.uz2);

	cudaFree(para.taoxx);
	cudaFree(para.taozz);
	cudaFree(para.taoxz);



	cudaFree(para.damp);

	cudaFree(para.record_ux);
	cudaFree(para.record_ux_mute);
	cudaFree(para.record_uz);
	cudaFree(para.record_uz_mute);

}
