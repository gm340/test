#pragma once
#include "cuda_runtime.h"
struct Parameter
{
	// host
	float* h_temp;

	// device
	float* vp0;
	float* vs0;

	float* ux0, * ux1, * ux2;
	float* uz0, * uz1, * uz2;


	float* taoxx, * taozz, * taoxz;



	float* record_ux;
    float* record_ux_mute;
	float* record_uz;
	float* record_uz_mute;

	float* damp; // PML coefficient
 
};

float* initializeArray(int row, int col, float value);
void** alloc2(size_t n1, size_t n2, size_t size);
int** alloc2int(size_t n1, size_t n2);
void index_shot_update(int min_shot, int max_shot, int** table, float dis_shot, float disx);
void readFile(char FNvelocity[], char FNvs0[], float* vp0, float* vs0, int Xn, int Zn, int L);
void ini_bdr(int Xn, int Zn, int L, float* ee);
void cuda_Device_malloc(int myid, struct Parameter& para, int Xn, int xn, int Zn, int zn, int NT);
void cuda_Device_free(struct Parameter& para);
void main_Forward(int myid, struct Parameter& para, int Xn, int  Zn, int  L, int Tn, int sxnum, int shotnum, int sy, int gy,
	float dx, float dz, float dt, float FM, float* vp0, float* vs0, float* record_ux, float* record_ux_mute, float* record_uz, float* record_uz_mute, int WriteSnapshot);
 