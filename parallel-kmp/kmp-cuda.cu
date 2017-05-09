/*
 * ECE 5720 Parallel Computing Final Project
 * KMP parallel on MPI
 * Feng Qi, fq26
 * Ying Zong, yz887
 * Cornell University
 *
 * Compile : /usr/local/cuda-8.0/bin/nvcc -arch=compute_35 -o cuda kmp-cuda.cu
 * Run     : ./cuda
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// build the kmp table for the subsequent operations
void preKMP(char* pattern, int func[]) {
    int m = strlen(pattern);
    int k;
    func[0] = -1;
    for (int i = 1; i < m; i++) {
        k = func[i - 1];
        while (k >= 0) {
            if (pattern[k] == pattern[i - 1]) {
                 break;
            }
            else {
                k = func[k];
            }
        }
        func[i] = k + 1;
    }
}

// Kernel function. Implement the KMP algorithm
__global__ void KMP(char* pattern, char* target, int func[], int answer[], int pattern_length, int target_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = pattern_length * index;
    int j = pattern_length * (index + 2) - 1;
    
    if(i > target_length) {
        return;
    }

    if(j > target_length) {
        j = target_length;
    }

    int k = 0;        
    while (i < j) {
        if (k == -1) {
            i++;
            k = 0;
        } else if (target[i] == pattern[k]) {
            i++;
            k++;
            if (k == pattern_length) {
                answer[i - pattern_length] = i - pattern_length;
                i = i - k + 1;
            }
        }
        else {
            k = func[k];
        }
    }
    return;
}

int main(int argc, char* argv[]) {
    const int target_length = 260;
    const int pattern_length = 4;
    int M = 4;

    char *target;
    char *pattern;
    target = (char*)malloc(target_length * sizeof(char));
    pattern = (char*)malloc(pattern_length * sizeof(char));

    char* dict = "abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < target_length; i++) {
        target[i] = dict[i%26];
    }
    for (int j = 0; j < pattern_length; j++) {
        pattern[j] = dict[j%26];
    }
    char *d_target;
    char *d_pattern;
    printf("----- This is parallel results using KMP Algo on CUDA. -----\n");
    // printf("The target length is: %d, the pattern length is %d.\n", target_length, pattern_length);
    // printf("The target string is: \n");
    // printf("%s\n", target);
    // printf("The pattern string is: \n");
    // printf("%s\n", pattern);
    int *func;
    int *answer;

    func = new int[target_length];
    answer = new int[target_length];

    int *d_func;
    int *d_answer;
    for(int i = 0; i < target_length; i++) {
        answer[i] = -1;
    }     

    preKMP(pattern, func);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate( &start ); 
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );

    cudaMalloc((void **)&d_target, target_length * sizeof(char));
    cudaMalloc((void **)&d_pattern, pattern_length * sizeof(char));
    cudaMalloc((void **)&d_func, target_length * sizeof(int));
    cudaMalloc((void **)&d_answer, target_length * sizeof(int));

    cudaMemcpy(d_target, target, target_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pattern, pattern, pattern_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_func, func, target_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_answer, answer, target_length * sizeof(int), cudaMemcpyHostToDevice);

    KMP<<<(target_length / pattern_length + M)/M, M>>>(d_pattern, d_target ,d_func, d_answer, pattern_length, target_length);

    cudaMemcpy(answer, d_answer, target_length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    printf("When the target length is %d, pattern length is %d, the elapsed time is %0.3f ms.\n", target_length, pattern_length, elapsedTime); 

    for(int i = 0; i < target_length; i++) {
        if (answer[i] != -1) {
            printf("Find a matching substring starting at: %d.\n", i);
        }
    }

    cudaFree(d_target); 
    cudaFree(d_pattern); 
    cudaFree(d_func); 
    cudaFree(d_answer);

    return 0;
}
