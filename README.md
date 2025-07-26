What is this repository for?
=

It is an inverse time migration method of elastic wave equation based on P- and S- wave decoupling.

Build Instructions
=

Before compiling, make sure you have the following:

⦁	GNU

⦁	MPI

⦁	CUDA

Key Steps
=

Due to the excessive file size of the seismic records, we have uploaded the forward modeling program(record) to GitHub. Users must first execute this program to generate the seismic data.

Project Structure
=

This project implements inverse time migration imaging of decoupling equations using CUDA and MPI.

record:
-

main.cpp:Main framework

2DForward_NewEqu:cuda kernel implementations

functions.h:Custom header file

segy.h:Segy header file

Makefile:Compilation instructions

para:Parameter setting

run.sh:Sample launch script

nodefile:Running of card settings

migration1:
-

main.cpp:Main framework

2DRTM_NewEqu:cuda kernel implementations

functions.h:Custom header file

segy.h:Segy header file

Makefile:Compilation instructions

para:Parameter setting

run.sh:Sample launch script

nodefile:Running of card settings

Installation from github
=

If you want to access the source code and potentially contribute. You should follow the following steps.

1.Download
  -
Download from the Github repository: green button "clone or download". Then, unzip it on your computer.

2.A reasonable set of input parameters are as follows(record):
  -
dx=7.5        &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;               #Grid spacing in x

dz=7.5        &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                     #Grid spacing in z

L=100         &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                    #PML thickness

Xn1=2001      &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                #Grid points in x

Zn1=501       &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                  #Grid points in z

dis_shot      &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                    #Shot spacing

min_shot      &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                  #First shot

max_shot      &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                 #The last shot

scale=1       &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                      # receiver density

dt=7e-4       &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                      # Sampling interval

t=7           &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                               #Total sampling time

FM            &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                        #Source central frequency

sy               &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                       #Shot depth

gy                   &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                   #Receiver depth

FNvelocity          &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;            #P-wave velocity model

FNvs0                 &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;             #S-wave velocity model

3.A reasonable set of input parameters are as follows(migration1):
  -
dx=7.5              &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;               #Grid spacing in x

dz=7.5               &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;              #Grid spacing in z

L=100                 &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;             #PML thickness

Xn1=2001            &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;          #Grid points in x

Zn1=501             &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;            #Grid points in z

dis_shot            &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;              #Shot spacing

min_shot              &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;          #First shot

max_shot             &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;          #The last shot

scale=1              &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;               # receiver density

FM                   &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                 #Source central frequency

sy                   &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                  #Shot depth

gy                   &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;                   #Receiver depth

FNvelocity           &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;           #P-wave velocity model

FNvs0                 &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;             #S-wave velocity model

FNrecord_ux            &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;     #x-component seismic records read

FNrecord_uz           &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;      #z-component seismic records read

4.Change the directory of data output in the code before running.
  -
5.Please change the nodefile and run according to the actual running memory.
  -
Run Instructions
=

To run the program, do as follows(The execution procedures are consistent between the two directories):

1.cd /                     #after the /   is the directory where the program is stored
   
2.make

3.sh run.sh


