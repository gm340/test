#pragma once
#include "cuda_runtime.h"
struct Parameter
{
	// host
	float* h_temp;

	// device
	float* vp0;
	float* vs0;

	float* upx0, * upx1, * upx2;
	float* upz0, * upz1, * upz2;
	float* usx0, * usx1, * usx2;
	float* usz0, * usz1, * usz2;
	float* ux, * uz;
	float* theta, * omega;
	float* duzdx, * duzdz, * duxdz, * duxdx;

	float* r_upx0, * r_upx1, * r_upx2;
	float* r_upz0, * r_upz1, * r_upz2;
	float* r_usx0, * r_usx1, * r_usx2;
	float* r_usz0, * r_usz1, * r_usz2;
	float* r_ux, * r_uz;
	float* r_theta, * r_omega;
	float* r_duzdx, * r_duzdz, * r_duxdz, * r_duxdx;

	float* upx_down, * upx_up, * upx_left, * upx_right;
	float* upz_down, * upz_up, * upz_left, * upz_right;

	float* r_upx_down, * r_upx_up, * r_upx_left, * r_upx_right;
	float* r_upz_down, * r_upz_up, * r_upz_left, * r_upz_right;
	float* r_usx_down, * r_usx_up, * r_usx_left, * r_usx_right;
	float* r_usz_down, * r_usz_up, * r_usz_left, * r_usz_right;



	float* record_ux;
	float* record_uz;

	float* damp; // PML coefficient

	float* ux_bdr;  // boundary wavefield
	float* uz_bdr;
	float* theta_bdr;
	float* omega_bdr;
	float* duzdx_bdr;
	float* duzdz_bdr;
	float* duxdz_bdr;
	float* duxdx_bdr;

	float* illumination_pp;
	float* migration_pp;
	float* migration_ps;
	float* migration_poynting_pp;
	float* migration_poynting_ps;

};

float* initializeArray(int row, int col, float value);
void** alloc2(size_t n1, size_t n2, size_t size);
int** alloc2int(size_t n1, size_t n2);
void laplace_filter(int adj, int Zn1, int Xn1, float* in, float* out);
void index_shot_update(int min_shot, int max_shot, int** table, float dis_shot, float disx);
int index_shot(char* fn, int* nt, float* dt, int* ns, int** table);
void readFile(char FNvelocity[], char FNvs0[], float* vp0, float* vs0, int Xn, int Zn, int L);
void read_shot_gather_su(char* fn, long long int pos, int ntr, int nt, float* dat, int* gc);
void ini_bdr(int Xn, int Zn, int L, float* ee);
void cuda_Device_malloc(int myid, struct Parameter& para, int Xn, int xn, int Zn, int zn, int NT);
void cuda_Device_free(struct Parameter& para);
void main_RTM(int myid, struct Parameter& para, int Xn, int  Zn, int  L, int Tn, int sxnum, int shotnum, int sy, int gy,
	float dx, float dz, float dt, float FM, float* vp0, float* vs0, float* record_ux, float* record_uz, float* illumination, float* migration_pp, float* migration_ps, int WriteSnapshot);
