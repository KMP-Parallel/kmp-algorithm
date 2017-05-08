/* ECE 5720 Parallel Final Project 
 * Substring Matching brute force sequential*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


#define BILLION 1000000000L
void bruteforce_sequential(char* s, char* t);

int main(int argc, char** argv){
	char* s = "AABAABAABAAAABAABAABAA";
	char* t = "ABA";
	struct timespec start1, end1;
	double diff;

	printf("-----This is sequential results using brute force method.-----\n");
	clock_gettime(CLOCK_MONOTONIC, &start1);
	bruteforce_sequential(s,t);
	clock_gettime(CLOCK_MONOTONIC, &end1);
	diff =(end1.tv_sec - start1.tv_sec)*BILLION + (end1.tv_nsec - start1.tv_nsec);
	printf("The execution time of sequential algo using brute force is %lf ns.\n", diff);

}

void bruteforce_sequential(char* s, char* t){
	int n = strlen(s);
	int m = strlen(t);
	int j;
	for(int i=0;i<n-m+1;i++){
		for(j=i;j<m+i;j++){
			if(s[j] != t[j-i]) break;
		}
		if(j == m+i) printf("this is matching %d\n", j-m);
	}
}
