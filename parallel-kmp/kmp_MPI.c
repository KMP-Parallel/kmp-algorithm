/*
 * ECE 5720 Parallel Computing Final Project
 * KMP parallel on MPI
 * Feng Qi, fq26
 * Ying Zong, yz887
 * Cornell University
 *
 * Compile : mpicc -std=gnu11 -o out kmp_MPI.c
 * Run 	   : mpirun -mca plm_rsh_no_tree_spawn 1 --hostfile hostfile -np 3 ./out 
 */

#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h> 

#define master 0
#define MAX(a,b) ((a) > (b) ? a : b)
#define MILLION 1000000L


int* kmp(char* target, char* pattern, int* table, int myrank);
int* preKMP(char* pattern, int len);
void print(char* arr, int id, int len);
void fillup(char** matrix, int rankid, char* msg, int len, int offset);
void freeDouble(char** matrix, int nproc);
void printMatrix(char **matrix, int nproc, int id, int length);
int* getRealIdx(int* answer, int len, int myrank, int x, int y, int* length);
void pinpoint(int* result, int* msg, int shortlen);


void pinpoint(int* result, int* msg, int shortlen){
	int j = 0;
	for(j=0; j<shortlen; j++){
		if(j > 0 && msg[j] == 0)
			break;
		else{
			result[msg[j]] = 1;
		}
	}
}

// get the exact idx(es) in each process
// return back an int pointer
int* getRealIdx(int* answer, int len, int myrank, int x, int y, int* length){
	int* package = (int *) calloc(len, sizeof(int));
	int index;
	int real_len = 0;
	int i;
	for(i=0;i < len; i++){
		if(i > 0 && answer[i] == 0)
			break;
		else{
			if(myrank == 0){
				index =  answer[i];
				package[real_len++] = index;
				// printf("This is REAL index: %d\n", index);
			}else{
				index = x * myrank - y + answer[i];
				package[real_len++] = index;
				// printf("This is REAL index: %d\n", index);
			}
			
		}
	}
	int* result = (int*)malloc((real_len) * sizeof(int));

	for(i =0 ; i<real_len; i++){
		if(i > 0 && package[i] == 0)
			break;
		else{
			result[i] = package[i];
		}
	}
	*length = real_len;
	free(package);
	return result;
}

void printMatrix(char **matrix, int nproc, int id, int length){
	printf("This matrix at %d row.\n", id);
	int i ,j;
	for(i=0 ;i < nproc; i++){
		if(i == id){
			for(j=0;j<length;j++){
				printf("%c", matrix[i][j]);
				break;
			}
			printf("\n");
		}
	}
	printf("\n");
}


// free two-pointer matrix
void freeDouble(char** matrix, int nproc){
	for (int i=0; i<nproc; ++i) {
		free(matrix[i]);
	}
	free(matrix);
}


void fillup(char** matrix, int rankid, char* msg, int len, int offset){
	int i;
	for(i=0;i<len;i++)
		matrix[rankid][offset + i] = msg[i];
}

// print out one pointer list
void print(char* arr, int id, int len){
	int j;
	printf("This is rank %d\n", id);
	for(j=0; j<len; j++){
		printf("%c", arr[j]);
	}
	printf("\n");
}

// key func kmp
int* kmp(char* target, char* pattern, int* table, int myrank){
	int n = strlen(target);
	int m = strlen(pattern);
	int* answer = (int*) calloc(n-m+1,sizeof(int));
	int j=0;
	int i=0;
	int index = 0;
	while(i<n){
		if(pattern[j]== target[i])
		{
			j++;
			i++;
		}
		if(j == m){
			// printf("this is matching %d.\n", i-j);
			answer[index++] = i-j;
			j = table[j-1];

		}else if(i < n && pattern[j] != target[i]){
			if(j!=0)
				j = table[j-1];
			else
				i++;
		}
	}
	return answer;
}


int main(int argc, char** argv){
	int i,j;
	int n = 260;
	int m = 4;

	char* target = (char*)malloc(n * sizeof(char));
	char* pattern = (char*)malloc(m * sizeof(char));
	char* b="abcdefghijklmnopqrstuvwxyz";
    for (i = 0; i < m; i++) {
        pattern[i] = b[i];
    }
    for (j = 0; j < n; j++) {
        target[j] = b[j%26];
    }


	int tag = 1;
	int tag2 = 2;
	int* kmp_table = preKMP(pattern, m);

	int myrank, nproc;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Status status;

	// prerequisite
	// nproc <= len(target)/(2 * len(pattern))
	int start_pos = n/nproc;
	int str_per_proc = n/nproc;
	int end = str_per_proc + n % nproc;

	char** matrix = (char**)malloc(nproc*sizeof(char*));
	for(i=0;i<nproc;i++){
		matrix[i] = (char*)malloc((end+m-1) * sizeof(char));
	}

    char send_msg[MAX(m-1, end)];
	char recv_msg[MAX(m-1, end)];
    char end_msg[MAX(m-1, end)];

    double start,stop;

    
	if(myrank == master){
		printf("----- This is parallel results using KMP Algo with MPI. -----\n");
		start = MPI_Wtime();
		for(i=1;i< nproc;i++){	
			if(i == nproc-1){
				strncpy(send_msg, target + i*start_pos, end);
				MPI_Send(send_msg, end, MPI_CHAR, i, tag, MPI_COMM_WORLD);
			}else{
				strncpy(send_msg, target + i*start_pos, str_per_proc);
				MPI_Send(send_msg, str_per_proc, MPI_CHAR, i, tag, MPI_COMM_WORLD);
			}
			
			
		}
		strncpy(send_msg, target + master * start_pos, str_per_proc);
		fillup(matrix, myrank, send_msg, str_per_proc, 0);

	// The last process
	}else if(myrank == nproc-1){
		MPI_Recv(recv_msg, end, MPI_CHAR, master, tag, MPI_COMM_WORLD, &status);

		fillup(matrix, myrank, recv_msg, end, m-1);
	}else{
		MPI_Recv(recv_msg, str_per_proc, MPI_CHAR, master, tag, MPI_COMM_WORLD, &status);	
		fillup(matrix, myrank, recv_msg, str_per_proc, m-1);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(myrank == nproc-1){
		MPI_Recv(end_msg, m-1, MPI_CHAR, myrank-1, tag, MPI_COMM_WORLD, &status);

		fillup(matrix, myrank, end_msg, m-1, 0);
		int* answer = kmp(matrix[myrank], pattern, kmp_table, myrank);
		
		int len;
		int* result = getRealIdx(answer, end-m+1, myrank, str_per_proc, m-1, &len);
		free(answer);

		MPI_Send(result, len, MPI_INT, master, tag2, MPI_COMM_WORLD);
		free(result);
	}else{
		strncpy(send_msg, target + str_per_proc-m + 1 + myrank * str_per_proc, m-1);
		MPI_Send(send_msg, m-1, MPI_CHAR, myrank+1, tag, MPI_COMM_WORLD);

		/* Processes other than master one 
		 * Re-recv more msg from the previous process whose rank is myrank-1
		 * kmp implementation
		 * Send the result back to master process
		 */
		if(myrank!=master){
			MPI_Recv(end_msg, m-1, MPI_CHAR, myrank-1, tag, MPI_COMM_WORLD, &status);
			fillup(matrix, myrank, end_msg, m-1, 0);
			// implement kmp to get the matching result
			int* answer = kmp(matrix[myrank], pattern, kmp_table, myrank);
			int len;
			int* result = getRealIdx(answer, end-m+1, myrank, str_per_proc, m-1, &len);
			free(answer);
			MPI_Send(result, len, MPI_INT, master, tag2, MPI_COMM_WORLD);
			free(result);
		}
		/* master proess
		 * do kmp implementation
		 * recv all the msg from other processes and merge it using 'pinpoint' func
		 * print out the result
		 */
		if(myrank == master){
			int* final_result = (int *)calloc(n, sizeof(int));
			int* recv = (int *)calloc(end+m-1, sizeof(int));
			int* answer = kmp(matrix[myrank], pattern, kmp_table, myrank);
			int len;
			int* result = getRealIdx(answer, end-m+1, myrank, str_per_proc, m-1, &len);
			pinpoint(final_result,result, len);
			free(answer);
			free(result);
			int j;
			for (j = 0; j < end+m-1; j++)
			{
				recv[j] = 0;
			}
			
			for(j = 1; j <nproc; j++){
				MPI_Recv(recv, end+m-1, MPI_INT, j, tag2, MPI_COMM_WORLD, &status);
				pinpoint(final_result,recv, end+m-1);
			}
			stop = MPI_Wtime();
			free(recv);
			printf("When the target length is %d, pattern length is %d, the elapsed time is %0.3f ms.\n", n, m, (stop-start) * MILLION); 
			for(j=0;j<n;j++){
				 if(final_result[j] == 1)
					printf("Find a matching substring starting at: %d.\n", j);
			}
			free(final_result);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	freeDouble(matrix, nproc);
	MPI_Finalize();
	free(target);
	free(pattern);
	return 0;
}

// KMP match table
// public to every process
int* preKMP(char* pattern, int len){
	int k = 0;
	int i = 1;
	int* table = (int*)malloc(len * sizeof(int));
	table[0] = k;
	for(i=1;i<len;i++){
		while(k > 0 && pattern[i-1] != pattern[i]){
			table[i] = k-1;
			k = table[i];
		}
		if(pattern[i] == pattern[k])
			k++;
		table[i] = k;
	}
	return table;
}
