export CUDA_LAUNCH_BLOCKING=1
time mpirun -np 39 --hostfile nodefile  ./2DForward para
#time mpirun -np 9  ./RC para
#mpirun -np 2 nvprof -o out6%q{OMPI_COMM_WORLD_RANK}.nvvp -f  ./GPU2DFDTD 2dsynrtmpar 
