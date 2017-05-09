#!/usr/bin/python
"""
Plz give a parameter for MPI code: the number of processes you want to set

"""
import os
import sys
import subprocess
import time
sequential_home = os.path.expanduser('sequential-kmp/')
parallel_home = os.path.expanduser('parallel-kmp/')


brute_force_home = os.path.expanduser('sequential-kmp/seq_brute_force.c')
sequential_kmp_home = os.path.expanduser('sequential-kmp/kmp_simple.c')
kmp_mpi_home = os.path.expanduser('parallel-kmp/kmp_MPI.c')
kmp_cuda_home = os.path.expanduser('parallel-kmp/kmp-cuda.cu')


def main():
	cmd1 = 'gcc -std=gnu11 -o ' + sequential_home + 'bf '+ brute_force_home
	os.system(cmd1)
	cmd2 = './' + sequential_home + 'bf'
	os.system(cmd2)


	cmd1 = 'gcc -std=gnu11 -o ' + sequential_home + 'seq '+ sequential_kmp_home
	os.system(cmd1)
	cmd2 = './' + sequential_home + 'seq'
	os.system(cmd2)


	cmd1 = 'mpicc -std=gnu11 -o ' + parallel_home + 'mpi '+ kmp_mpi_home
	os.system(cmd1)
	cmd2 = 'mpirun -mca plm_rsh_no_tree_spawn 1 --hostfile ' + parallel_home + 'hostfile -np ' + \
			sys.argv[1] + ' ./' + parallel_home + 'mpi'
	os.system(cmd2)

	cmd1 = '/usr/local/cuda-8.0/bin/nvcc -arch=compute_35 -o ' + parallel_home + 'cuda '+ kmp_cuda_home
	os.system(cmd1)

	cmd2 = './' + parallel_home + 'cuda'
	os.system(cmd2)


if __name__ == '__main__':
	main()