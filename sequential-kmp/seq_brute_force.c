/*
 * ECE 5720 Parallel Computing Final Project
 * KMP parallel on MPI
 * Feng Qi, fq26
 * Ying Zong, yz887
 * Cornell University
 *
 * Compile : gcc -std=gnu11 -o out seq_brute_force.c
 * Run 	   : ./out 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MILLION 1000000L
void bruteforce_sequential(char* s, char* t);

int main(int argc, char** argv){
	int n = 100000;
	int m = 3; 
	int i,j;
	char* target = (char*)malloc(n * sizeof(char));
	char* pattern = (char*)malloc(m * sizeof(char));
	FILE * file = fopen( "data.txt" , "r");
	int CurrentIndex = 0;

    while (CurrentIndex < 2) {
        if (CurrentIndex == 0) {
            fgets(target, n+1, file);
        } else if (CurrentIndex == 1) {
            fgets(pattern, m+1, file);
        }
        CurrentIndex++;
    }
    fclose(file);
    printf("This is pattern: %s \n\n", pattern);
	struct timespec start1, end1;
	double diff;

	printf("----- This is sequential results using brute force method. -----\n");
	clock_gettime(CLOCK_MONOTONIC, &start1);
	bruteforce_sequential(target,pattern);
	clock_gettime(CLOCK_MONOTONIC, &end1);
	diff =(end1.tv_sec - start1.tv_sec)*MILLION + (end1.tv_nsec - start1.tv_nsec);
	printf("When the target length is %d, pattern length is %d, the elapsed time is %0.3f ms.\n", n, m, diff); 
	free(target);
	free(pattern);
	printf("\n");
	return 1;
}

void bruteforce_sequential(char* s, char* t){
	int n = strlen(s);
	int m = strlen(t);
	int i,j;
	for(i=0;i<n-m+1;i++){
		for(j=i;j<m+i;j++){
			if(s[j] != t[j-i]) break;
		}
		if(j == m+i) printf("Find a matching substring starting at: %d.\n", j-m);
	}
}
