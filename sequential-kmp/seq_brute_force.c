/* ECE 5720 Parallel Final Project 
 * Feng Qi, Ying Zong
 * Substring Matching brute force sequential 
 * Compile:
 * gcc -std=gnu11 -o ref seq_brute_force.c
 * ./ref
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


#define MILLION 1000000L
void bruteforce_sequential(char* s, char* t);

int main(int argc, char** argv){

	// char* s = "AABAABAABAAAABAABAABAA";
	// char* t = "ABA";
	int n = 260;
	int m = 4; 
	int i,j;
	char* target = (char*)malloc(n * sizeof(char));
	char* pattern = (char*)malloc(m * sizeof(char));
	char* b="abcdefghijklmnopqrstuvwxyz";
    for (i = 0; i < n; i++) {
        target[i] = b[i%26];
    }
    for (j = 0; j < m; j++) {
        pattern[j] = b[j];
    }
	struct timespec start1, end1;
	double diff;

	printf("-----This is sequential results using brute force method.-----\n");
	clock_gettime(CLOCK_MONOTONIC, &start1);
	bruteforce_sequential(target,pattern);
	clock_gettime(CLOCK_MONOTONIC, &end1);
	diff =(end1.tv_sec - start1.tv_sec)*MILLION + (end1.tv_nsec - start1.tv_nsec);
	printf("The execution time of sequential algo using brute force is %.3f ms.\n", diff);
	free(target);
	free(pattern);
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
		if(j == m+i) printf("this is matching %d\n", j-m);
	}
}
