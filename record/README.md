#### What is this repository for?

###### It is an inverse time migration method of elastic wave equation based on P- and S- wave decoupling.



#### Build Instructions

###### Before compiling, make sure you have the following:

* ###### GNU
* ###### MPI
* ###### CUDA



#### Key Steps

###### Due to the excessive file size of the seismic records, we have uploaded the forward modeling program(record) to GitHub. Users must first execute this program to generate the seismic data.



#### Project Structure

###### This project implements inverse time migration imaging of decoupling equations using CUDA and MPI.



##### record:

###### main.cpp:Main framework

###### 2DForward\_NewEqu:cuda kernel implementations

###### functions.h:Custom header file

###### segy.h:Segy header file

###### Makefile:Compilation instructions

###### para:Parameter setting

###### run.sh:Sample launch script

###### nodefile:Running of card settings



##### migration1:

###### main.cpp:Main framework

###### 2DRTM\_NewEqu:cuda kernel implementations

###### functions.h:Custom header file

###### segy.h:Segy header file

###### Makefile:Compilation instructions

###### para:Parameter setting

###### run.sh:Sample launch script

###### nodefile:Running of card settings

#### Installation from github

###### If you want to access the source code and potentially contribute. You should follow the following steps.

###### 

###### 1\. Download

###### Download from the Github repository: green button "clone or download". Then, unzip it on your computer.



###### 2.A reasonable set of input parameters are as follows(record):

###### dx=7.5                           #Grid spacing in x

###### dz=7.5                           #Grid spacing in z

###### L=100                            #PML thickness

###### Xn1=2001                    #Grid points in x

###### Zn1=501                       #Grid points in z

###### dis\_shot                        #Shot spacing

###### min\_shot                      #First shot

###### max\_shot                     #The last shot

###### scale=1                           # receiver density

###### dt=7e-4                           # Sampling interval

###### t=7                                        #Total sampling time

###### FM                                  #Source central frequency

###### sy                                    #Shot depth

###### gy                                    #Receiver depth

###### FNvelocity                    #P-wave velocity model

###### FNvs0                            #S-wave velocity model



###### 3.A reasonable set of input parameters are as follows(migration1):

###### dx=7.5                           #Grid spacing in x

###### dz=7.5                           #Grid spacing in z

###### L=100                            #PML thickness

###### Xn1=2001                    #Grid points in x

###### Zn1=501                       #Grid points in z

###### dis\_shot                        #Shot spacing

###### min\_shot                      #First shot

###### max\_shot                     #The last shot

###### scale=1                           # receiver density

###### FM                                  #Source central frequency

###### sy                                    #Shot depth

###### gy                                    #Receiver depth

###### FNvelocity                    #P-wave velocity model

###### FNvs0                            #S-wave velocity model

###### FNrecord\_ux               #x-component seismic records read

###### FNrecord\_uz               #z-component seismic records read

###### 3.Change the directory of data output in the code before running.

###### 4.Please change the nodefile and run according to the actual running memory.



### Run Instructions

###### To run the program, do as follows(The execution procedures are consistent between the two directories):

1. ###### cd /                     #after the /   is the directory where the program is stored

###### 2.make

###### 3.sh run.sh



###### 

