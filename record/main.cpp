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

	int Xn1, Zn1, Tn, L;
	int ns, min_shot, max_shot;
	float dis_shot, disx;
	int scale,sy, gy;
	int i, j;
	clock_t time, caltime;
	float dx, dz, dt, t, FM;

	char filename[1024];

	char FNvelocity[1024] ;
	char FNvs0[1024] ;

	char FNrecord_ux[1024];
 	char FNrecord_ux_mute[1024];
	char FNrecord_uz[1024];
	char FNrecord_uz_mute[1024];
 
        int WriteSnapshot = 1; 

	strcpy(filename, argv[1]);
	FILE* fp = NULL;

	fp = fopen(filename, "r");
	fscanf(fp, "%f %f %d %d %d ", &dx, &dz, &L, &Xn1, &Zn1);
	fscanf(fp, "%f %d %d", &dis_shot, &min_shot, &max_shot);          
	fflush(stdout);

	fscanf(fp, "%d", &scale);  
	fscanf(fp, "%f %f %f", &dt, &t, &FM);
	fscanf(fp, "%d %d", &sy, &gy);


	fscanf(fp, "%s", FNvelocity);
	fscanf(fp, "%s", FNvs0);
	fclose(fp);

	int xtrace;
	
	Tn = ceil(t / dt) + 1;
	xtrace = Xn1 / scale;

	int Xn = Xn1 + 2 * L;
	int Zn = Zn1 + 2 * L;
	sy += L;
	gy += L;
	ns = max_shot - min_shot + 1;

	int** table = NULL;
	table = alloc2int(4, ns);

	index_shot_update(min_shot, max_shot, table, dis_shot, disx);

	// segy head
	segy Th;

	fflush(stdout);
	float* h_vp0 = initializeArray(Zn, Xn, 0.0);
	float* h_vs0 = initializeArray(Zn, Xn, 0.0);

	float* h_record_ux = initializeArray(Tn, Xn1, 0.0);
 	float* h_record_ux_mute = initializeArray(Tn, Xn1, 0.0);
	float* h_record_uz = initializeArray(Tn, Xn1, 0.0);
	float* h_record_uz_mute = initializeArray(Tn, Xn1, 0.0);

	if (myid == 0)
	{
		readFile(FNvelocity, FNvs0, h_vp0, h_vs0, Xn, Zn, L);
		ini_bdr(Xn, Zn, L, h_vp0);
		ini_bdr(Xn, Zn, L, h_vs0);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(h_vp0, Xn * Zn, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(h_vs0, Xn * Zn, MPI_FLOAT, 0, MPI_COMM_WORLD);

	int ip;
	int send[4], recv[4];
	int nsend;

	sprintf(FNrecord_ux, "/public/home/scw/Desktop/2023413/displacement_stress/record/record_ux_%d_%d_ns%d.su", Xn1, Tn, ns);
	FILE* frecord_ux = NULL;
	frecord_ux = fopen(FNrecord_ux, "wb");

	sprintf(FNrecord_ux_mute, "/public/home/scw/Desktop/2023413/displacement_stress/record/record_ux_mute_%d_%d_ns%d.su", Xn1, Tn, ns);
	FILE* frecord_ux_mute = NULL;
	frecord_ux_mute = fopen(FNrecord_ux_mute, "wb");

	sprintf(FNrecord_uz, "/public/home/scw/Desktop/2023413/displacement_stress/record/record_uz_%d_%d_ns%d.su", Xn1, Tn, ns);
	FILE* frecord_uz = NULL;
	frecord_uz = fopen(FNrecord_uz, "wb");

	sprintf(FNrecord_uz_mute, "/public/home/scw/Desktop/2023413/displacement_stress/record/record_uz_mute_%d_%d_ns%d.su", Xn1, Tn, ns);
	FILE* frecord_uz_mute = NULL;
	frecord_uz_mute = fopen(FNrecord_uz_mute, "wb");

 
	MPI_Barrier(MPI_COMM_WORLD);
	Parameter para1 ;

	if (myid == 0) 
	{
		nsend = 0;
		for (i = 0; i < ns + np - 1; i++)
		{
			MPI_Recv(recv, 4, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			ip = status.MPI_SOURCE;
			if (i < ns)
			{
				send[0] = table[i][0];             
				send[1] = table[i][1];		     
				send[2] = table[i][2];		    
				send[3] = table[i][3];		  
			}
			else
			{
				send[0] = 0;
			}

			MPI_Send(send, 4, MPI_INT, ip, 99, MPI_COMM_WORLD);
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

		MPI_Send(send, 4, MPI_INT, 0, 0, MPI_COMM_WORLD);
		for (;;) 
		{
			MPI_Recv(recv, 4, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);		

			int shotnum = recv[0];
			float sourceloc = recv[1];
			float gx_min = recv[2];
			float gx_max = recv[3];

			if (shotnum == 0)
			{
				printf("Processor %d has finished task.\n", myid);
				break;
			}
			int ngx_min = gx_min / dx;
			int sxnum;
			sxnum = (sourceloc - gx_min) / dx;
			sxnum += L;

			
		
		
			for (i = 0; i < xtrace * Tn; i++) { h_record_ux[i] = 0; }
			for (i = 0; i < xtrace * Tn; i++) { h_record_ux_mute[i] = 0; }
			for (i = 0; i < xtrace * Tn; i++) { h_record_uz[i] = 0; }
			for (i = 0; i < xtrace * Tn; i++) { h_record_uz_mute[i] = 0; }

			main_Forward(myid, para1, Xn, Zn, L, Tn, sxnum, shotnum, sy, gy, dx, dz, dt, FM, h_vp0, h_vs0, h_record_ux, h_record_ux_mute, h_record_uz, h_record_uz_mute, WriteSnapshot);

			long int offsett = (240 + sizeof(float) * Tn) * xtrace * (shotnum - 1);
			int numm = offsett / (240 + 4 * Tn) / xtrace;
			
      
			fseek(frecord_ux, offsett, SEEK_SET);
			fseek(frecord_ux_mute, offsett, SEEK_SET);

			fseek(frecord_uz, offsett, SEEK_SET);
			fseek(frecord_uz_mute, offsett, SEEK_SET);
      
			short  nss;
			short  hdt;
			nss = Tn;
			hdt = dt * 1000000;
			memset(&Th, 0, sizeof(segy));
			for (int ii = 0; ii < xtrace; ii++)
			{
				Th.tracl = ii + 1;        

				Th.fldr = shotnum;           

				Th.ep = shotnum; 
			                         
				Th.sx = sourceloc;          
				                            
				Th.gx = ii * dx * scale + gx_min;
			    
				Th.offset = Th.gx - Th.sx;
				Th.ns = nss;       
				Th.dt = hdt;        
			 

				fwrite(&Th, 240, 1, frecord_ux);
				fwrite(&h_record_ux[ii * Tn], sizeof(float) * Tn, 1, frecord_ux);
        
				fwrite(&Th, 240, 1, frecord_ux_mute);
				fwrite(&h_record_ux_mute[ii * Tn], sizeof(float) * Tn, 1, frecord_ux_mute);

				fwrite(&Th, 240, 1, frecord_uz);
				fwrite(&h_record_uz[ii * Tn], sizeof(float) * Tn, 1, frecord_uz);

				fwrite(&Th, 240, 1, frecord_uz_mute);
				fwrite(&h_record_uz_mute[ii * Tn], sizeof(float) * Tn, 1, frecord_uz_mute);
			}

			MPI_Send(send, 4, MPI_INT, 0, myid, MPI_COMM_WORLD);
		}

		cuda_Device_free(para1);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	fclose(frecord_ux);
 	fclose(frecord_ux_mute);
	fclose(frecord_uz);
	fclose(frecord_uz_mute);


	free(h_vp0);
	free(h_vs0);

	free(h_record_ux);
 	free(h_record_ux_mute);
	free(h_record_uz);
	free(h_record_uz_mute);

	MPI_Finalize();
}
