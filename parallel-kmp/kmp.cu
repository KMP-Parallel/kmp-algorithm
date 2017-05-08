#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


void preKMP(char* pattern, int f[])
{
    int m = strlen(pattern), k;
    f[0] = -1;
    for (int i = 1; i < m; i++)
    {
        k = f[i - 1];
        while (k >= 0)
        {
            if (pattern[k] == pattern[i - 1])
                break;
            else
                k = f[k];
        }
        f[i] = k + 1;
    }
}

//check whether target string contains pattern 
__global__ void KMP(char* pattern, char* target,int f[],int c[],int n, int m)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int i = n * index;
    int j = n * (index + 2)-1;
    if(i>m)
        return;
    if(j>m)
        j=m;
    int k = 0;        
    while (i < j)
    {
        if (k == -1)
        {
            i++;
            k = 0;
        }
        else if (target[i] == pattern[k])
        {
            i++;
            k++;
            if (k == n)
            {
                c[i - n] = i-n;
                i = i - k + 1;
            }
        }
        else
            k = f[k];
    }
    return;
}

int main(int argc, char* argv[])
{
    const int L = 260;
    const int S = 4;
    int M = 4;//num of threads

    char *tar;
    char *pat;
    tar = (char*)malloc(L*sizeof(char));
    pat = (char*)malloc(S*sizeof(char));

    char* b="abcdefghijklmnopqrstuvwxyz";
    for (int i = 0; i < L; i++) {
        tar[i] = b[i%26];
    }
    for (int j=0; j < S; j++) {
        pat[j] = b[j];
    }
    char *d_tar;
    char *d_pat;

    int m = strlen(tar);
    int n = strlen(pat);
    printf("%d %d\n",m,n);
    int *f;
    int *c;

    f = new int[m];
    c = new int[m];

    int *d_f;
    int *d_c;
    for(int i = 0;i<m; i++)
    {
        c[i] = -1;
    }     

    preKMP(pat, f);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate( &start ); 
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 );

    cudaMalloc((void **)&d_tar, m*sizeof(char));
    cudaMalloc((void **)&d_pat, n*sizeof(char));
    cudaMalloc((void **)&d_f, m*sizeof(int));
    cudaMalloc((void **)&d_c, m*sizeof(int));

    cudaMemcpy(d_tar, tar, m*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat, pat, n*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, m*sizeof(int), cudaMemcpyHostToDevice);

    KMP<<<(m/n+M)/M,M>>>(d_pat, d_tar ,d_f, d_c, n, m);

    cudaMemcpy(c, d_c, m*sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    printf("When the target length is %d, pattern length is %d, the elapsed time is %0.3f.\n ms", m, n, elapsedTime); 


    for(int i = 0; i < m; i++) {
        //printf("%d\n", c[i]);
    }
    cudaFree(d_tar); cudaFree(d_pat); cudaFree(d_f); cudaFree(d_c);
    return 0;
}}