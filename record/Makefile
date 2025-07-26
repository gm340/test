#! /bin/sh

# compiler
NVCC=nvcc
CC=mpicxx
# set CUDA environment path 
INC=-I/usr/local/cuda-9.0/include 
LIB=-L/usr/local/cuda-9.0/lib64 
# set CUDA and link library
LINK= -lcudart -lcufft 

# CUDA and C++ source codes
SOURCES1=2DForward_NewEqu.cu
SOURCES2=main.cpp
EXECNAME=2DForward

# Execution
all:
	$(NVCC) -c $(SOURCES1) $(INC) $(LIB) -w -lcudart -lcufft 
	$(CC) -c  $(SOURCES2) $(INC) $(LIB) 
	$(CC) -o $(EXECNAME)  main.o 2DForward_NewEqu.o  $(INC) $(LIB) -lcudart -lcufft
clean:
	rm -f *.o 
