#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "functions.h"
#include<iostream>
#include"segy.h"
using namespace std;
#define PI 3.1415926

int main(int argc, char* argv[])
{

	int myid, np;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	if (myid == 0)
	{
		clock_t time, caltime;
		time = clock();
	}

	int Xn1, Zn1, Tn, L;
	int ns, min_shot, max_shot;
	float dis_shot, disx;//Num_SumShots & Miniest shot & Maxest shot & Shot spacing(m) & Receive leghth(m)
	int scale, sy, gy;// Receive denisty & mode==1 represents Fixed acceptance, mode==2 represents Mobile acceptance & Depth of source & Depth of receiver
	int i, j;
	clock_t time, caltime;
	float dx, dz, dt, t, FM;// Grid spacing & Time step & Real time & Wavelet domain frequency

	char filename[1024];// General filename

	char FNvelocity[1024];
	char FNvs0[1024];

	char FNrecord_ux[1024];
	char FNrecord_uz[1024];
 
  int WriteSnapshot = 0; // 1 for Write

	strcpy(filename, argv[1]);// .exe para
	FILE* fp = NULL;
	fp = fopen(filename, "r");
	fscanf(fp, "%f %f %d %d %d ", &dx, &dz, &L, &Xn1, &Zn1);
	fscanf(fp, "%f %d %d", &dis_shot, &min_shot, &max_shot);       //dis_shot(m)
	fflush(stdout);
	//	printf("dishot:%f,minshot:%d,maxsoht:%d\n",dis_shot,min_shot,max_shot);
	fscanf(fp, "%d", &scale);    //receiver density  
	//fscanf(fp, "%f %f %f", &dt, &t, &FM);//read dt&t
	fscanf(fp, "%f", &FM);
	fscanf(fp, "%d %d", &sy, &gy);

	//printf("scale:%d,mode:%d,dt:%f,t:%f,f0:%f,sy:%d,gy:%d,dx:%f,dz:%f\n", scale, mode,dt,t,f0,sy,gy,dx,dz);
	fscanf(fp, "%s", FNvelocity);
	fscanf(fp, "%s", FNvs0);
	fscanf(fp, "%s", FNrecord_ux);
	fscanf(fp, "%s", FNrecord_uz);
	fclose(fp);

	int xtrace;
	//(nx%scale==0)?xtrace=nx/scale:xtrace=nx/scale+1;
	//Tn = ceil(t / dt) + 1; 
	xtrace = Xn1 / scale;

	int Xn = Xn1 + 2 * L;
	int Zn = Zn1 + 2 * L;
	sy += L;
	gy += L;
	//ns = max_shot - min_shot + 1;//read ns
	float x0 = 0.0f;
	float cmin, cmax, cmleft, cmright;
	cmin = x0;
	cmax = x0 + (Xn - 1) * dx;

	int** table = NULL;
	table = alloc2int(9, 10000);//10000 can be repalced by maxshot;

	if (myid == 0)
	{
		if (index_shot(FNrecord_ux, &Tn, &dt, &ns, table))
		{
			printf("Can not read the shot file!\n");
			return 0;
		}
		cmleft = 999999;
		cmright = -999999;
		for (i = 0; i < ns; i++)
		{
			if (cmleft > table[i][4])cmleft = table[i][4];
			if (cmright < table[i][6])cmright = table[i][6];
			if (cmin > table[i][4])printf(" Warning! The %dth Shot's minimum coordinate is on the left of velocity model\n", i);
			if (cmax < table[i][6])printf(" Warning! The %dth Shot's maximum coordinate is on the right of velocity model\n", i);
		}
	}
	if (myid == 0)
	{
		printf("==========Parameters of input seismic file============\n");
		printf(" Shot number           : %d\n", ns);
		printf(" Sampling point number : %d\n", Tn);
		printf(" Sampling interval     : %f s\n", dt);
		printf("======================================================\n\n");
		//   for(i=0;i<ns;i++)printf(" table[%d][8] : %d\n",i,table[i][8]);
	}

	int ntr_pre, nx, nz;
	if (myid == 0)
	{
		ntr_pre = table[0][1];
		nx = (int)(table[0][6] / dx + 0.5) - (int)(table[0][4] / dx + 0.5) + 1;
		nz = Zn1;
		printf("nx=%d,nz=%d\n", nx, nz);
	}

	MPI_Bcast(&ns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Tn, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ntr_pre, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nz, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);

	fflush(stdout);
	float* h_vp0 = initializeArray(Zn, Xn, 0.0);
	float* h_vs0 = initializeArray(Zn, Xn, 0.0);

	float* h_illumination = initializeArray(Zn1, Xn1, 0);
	float* h_migration_pp = initializeArray(Zn1, Xn1, 0);
	float* h_migration_pp_stack = initializeArray(Zn1, Xn1, 0);
	float* h_migration_pp_la = initializeArray(Zn1, Xn1, 0);

	float* h_migration_ps = initializeArray(Zn1, Xn1, 0);
	float* h_migration_ps_stack = initializeArray(Zn1, Xn1, 0);
	float* h_migration_ps_la = initializeArray(Zn1, Xn1, 0);

	float* h_record_ux = initializeArray(Tn, xtrace, 0.0);
	float* h_record_uz = initializeArray(Tn, xtrace, 0.0);

	if (myid == 0)
	{
		readFile(FNvelocity, FNvs0, h_vp0, h_vs0, Xn, Zn, L);
		ini_bdr(Xn, Zn, L, h_vp0);
		ini_bdr(Xn, Zn, L, h_vs0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(h_vp0, Xn * Zn, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_vs0, Xn * Zn, MPI_FLOAT, 0, MPI_COMM_WORLD);

	int* gc = new int[ntr_pre];
	int ip;
	int send[9], recv[9];
	int nsend, ntask;
	ntask = ns;

	MPI_Barrier(MPI_COMM_WORLD);
	Parameter para1;

	if (myid == 0)
	{
		caltime = clock();
	}

	if (myid == 0)
	{
		nsend = 0;
		for (i = 0; i < ntask + np - 1; i++)
		{
			MPI_Recv(recv, 9, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			ip = status.MPI_SOURCE;
			if (i < ns)
			{
				send[0] = table[i][0];               //shot number
				send[1] = table[i][1];		      // ntr
				send[2] = table[i][2];		      // source x
				send[3] = table[i][3];		    
				send[4] = table[i][4];
				send[5] = table[i][5];
				send[6] = table[i][6];
				send[7] = table[i][7];
				send[8] = table[i][8];
			}
			else
			{
				send[0] = 0;
			}

			MPI_Send(send, 9, MPI_INT, ip, 99, MPI_COMM_WORLD);
			nsend = nsend + 1;
			if (send[0] > 0)
			{
				printf("Doing send %d.  Shot No.=%d to Processor %d\n", nsend, send[0], ip);
			}
		}
	}
	else
	{
		cudaSetDevice((myid - 1) % 8);
		cuda_Device_malloc(myid, para1, Xn, Xn1, Zn, Zn1, Tn);

		MPI_Send(send, 9, MPI_INT, 0, 0, MPI_COMM_WORLD);
		for (;;)
		{
			MPI_Recv(recv, 9, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);

			//cuda_Device_malloc(myid,A, SF, SC, m, Xn, xn, Zn, nt);				
			int shotnum = recv[0];
			int ntr = recv[1];
			int sourcelocx = recv[2];//Horizontal position fo current shot (m)
			int sourcelocy = recv[3];//Horizontal position fo current shot (m)
			int cx_min_s = recv[4];
			int cy_min_s = recv[5];
			int cx_max_s = recv[6];
			int cy_max_s = recv[7];
			int pos = recv[8];

			if (shotnum == 0)
			{
				fflush(stdout);
				printf("Processor %d has finished task.\n", myid);
				break;
			}
			if (shotnum <min_shot || shotnum>max_shot + 1)
			{
				fflush(stdout);
				printf("The %dth shot is out of image range.\n", shotnum);
				MPI_Send(send, 9, MPI_INT, 0, myid, MPI_COMM_WORLD);
				continue;
			}
			int ngx_left = (int)(cx_min_s / dx + 0.5);
			int ngx_right = (int)(cx_max_s / dx + 0.5);
			int nx_s = ngx_right - ngx_left + 1;

			memset(h_record_ux, 0, sizeof(float) * xtrace * Tn);
			memset(h_record_uz, 0, sizeof(float) * xtrace * Tn);

			if (ntr == ntr_pre)
			{
				read_shot_gather_su(FNrecord_ux, pos, ntr, Tn, h_record_ux, gc);
				read_shot_gather_su(FNrecord_uz, pos, ntr, Tn, h_record_uz, gc);
				if (shotnum == 1)
				{
					sprintf(filename, "record_%d.dat", shotnum);
					if ((fp = fopen(filename, "wb")) != NULL)
					{
						for (int i = 0; i < ntr; i++)
						{
							fwrite(&h_record_ux[i * Tn], sizeof(float) * Tn, 1, fp);
							fwrite(&h_record_uz[i * Tn], sizeof(float) * Tn, 1, fp);
						}
						fclose(fp);
					}
				}
			}
			else
			{
				printf("Trace numbers differ from preload ntr_pre in this shot. Need reallocate.\n");
				delete[] h_record_ux;
				delete[] h_record_uz;
				delete[] gc;
				float* h_record_ux = new float[Tn * ntr];
				float* h_record_uz = new float[Tn * ntr];
				int* gc = new int[ntr];
				read_shot_gather_su(FNrecord_ux, pos, ntr, Tn, h_record_ux, gc);
				read_shot_gather_su(FNrecord_uz, pos, ntr, Tn, h_record_uz, gc);
			}


			int sxnum, sznum;//Horizontal grid position of current shot
			sxnum = (sourcelocx - ngx_left) / dx + L;
			int ii = 0;
			int ngx_min = ngx_left / dx;
			printf("sxnum=%d,ngx_min=%d\n", sxnum - L, ngx_min);

			memset(h_migration_pp, 0, sizeof(float)* Xn1* Zn1);
			memset(h_migration_ps, 0, sizeof(float)* Xn1* Zn1);

			main_RTM(myid, para1, Xn, Zn, L, Tn, sxnum, shotnum, sy, gy, dx, dz, dt, FM, h_vp0, h_vs0, h_record_ux, h_record_uz, h_illumination, h_migration_pp, h_migration_ps, WriteSnapshot);

			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/migration_pp/pp_migration_%d_%d_%d_ns%d.dat", Xn1, Zn1, shotnum, ns);
			if ((fp = fopen(filename, "wb")) != NULL)
			{
				for (i = 0; i < Xn1; i++)
				{
					fwrite(&h_migration_pp[i * Zn1], Zn1 * sizeof(float), 1, fp);
				}
				fclose(fp);
			}

			sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/migration_ps/ps_migration_%d_%d_%d_ns%d.dat", Xn1, Zn1, shotnum, ns);
			if ((fp = fopen(filename, "wb")) != NULL)
			{
				for (i = 0; i < Xn1; i++)
				{
					fwrite(&h_migration_ps[i * Zn1], Zn1 * sizeof(float), 1, fp);
				}
				fclose(fp);
			}

			for (i = 0; i < Xn1; i++)
			{
				for (j = 0; j < Zn1; j++)
				{
					h_migration_pp_stack[i * Zn1 + j] += h_migration_pp[i * Zn1 + j];
					h_migration_ps_stack[i * Zn1 + j] += h_migration_ps[i * Zn1 + j];
				}
			}

			MPI_Send(send, 9, MPI_INT, 0, myid, MPI_COMM_WORLD);
		}

		cuda_Device_free(para1);
	} // process

	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

	if (myid == 0)
	{
		MPI_Reduce(MPI_IN_PLACE, &h_migration_pp_stack[0], Xn1 * Zn1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE, &h_migration_ps_stack[0], Xn1 * Zn1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Reduce(&h_migration_pp_stack[0], &h_migration_pp_stack[0], Xn1 * Zn1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&h_migration_ps_stack[0], &h_migration_ps_stack[0], Xn1* Zn1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	}
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);


	if (myid == 0)
	{
		sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/migration_pp/pp_migration_stack_%d_%d_ns%d.dat",Xn1,Zn1,ns);
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = 0; i < Xn1; i++)
			{
				fwrite(&h_migration_pp_stack[i * Zn1], Zn1 * sizeof(float), 1, fp);
			}
			fclose(fp);
		}

		sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/migration_ps/ps_migration_stack_%d_%d_ns%d.dat", Xn1, Zn1, ns);
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = 0; i < Xn1; i++)
			{
				fwrite(&h_migration_ps_stack[i * Zn1], Zn1 * sizeof(float), 1, fp);
			}
			fclose(fp);
		}

		///poststack laplace
		laplace_filter(1, Zn1, Xn1, h_migration_pp_stack, h_migration_pp_la);
		laplace_filter(1, Zn1, Xn1, h_migration_ps_stack, h_migration_ps_la);

		sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/migration_pp/pp_migration_stack_la_%d_%d_ns%d.dat",Xn1,Zn1,ns);
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = 0; i < Xn1; i++)
			{
				fwrite(&h_migration_pp_la[i * Zn1], Zn1 * sizeof(float), 1, fp);
			}
			fclose(fp);
		}

		sprintf(filename, "/public/home/scw/Desktop/2023413/newrtm_poynting/migration_ps/ps_migration_stack_la_%d_%d_ns%d.dat", Xn1, Zn1, ns);
		if ((fp = fopen(filename, "wb")) != NULL)
		{
			for (i = 0; i < Xn1; i++)
			{
				fwrite(&h_migration_ps_la[i * Zn1], Zn1 * sizeof(float), 1, fp);
			}
			fclose(fp);
		}

	}

	if (myid == 0)
	{
		caltime = clock() - caltime;
		printf("Total calculation time = %f (s)\n", ((float)caltime) / CLOCKS_PER_SEC);
	}

	free(h_vp0);
	free(h_vs0);


	free(h_illumination);
	free(h_migration_pp);
	free(h_migration_pp_stack);
	free(h_migration_pp_la);
	free(h_migration_ps);
	free(h_migration_ps_stack);
	free(h_migration_ps_la);

	free(h_record_ux);
	free(h_record_uz);

	if (myid == 0)
	{
		time = clock() - time;
		printf("End!!! Total time = %f (s)\n", ((float)time) / CLOCKS_PER_SEC);
	}

	MPI_Finalize();
}
