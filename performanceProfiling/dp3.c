#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include <mkl_cblas.h>


float bdp(long N, float *pA, float *pB) {
  float R = cblas_sdot(N, pA, 1, pB, 1);
  return R;
}

int main(int argc,char **argv){
	
	long length_of_array = atol(argv[1]);
    int loops_reps = atoi(argv[2]);
	
	float *x = (float *)malloc(sizeof(float) * length_of_array);
	float *y = (float *)malloc(sizeof(float) * length_of_array);
	
	int i;
	
	
	for (i=0;i<length_of_array;i++){
		x[i]=1.0;
		y[i]=1.0;
	}
	
	float finalans = 0.0;
	struct timespec start, end;
	float timetaken = 0.0;
	
	for (i=1;i<=loops_reps;i++){
		clock_gettime(CLOCK_MONOTONIC, &start);
		finalans = bdp(length_of_array, x, y);
		clock_gettime(CLOCK_MONOTONIC, &end);
		int m1 = 1000000;
		int m2 = 1000;
		float time_usec=(((float)end.tv_sec - (float)start.tv_sec) *m1) + (((float)end.tv_nsec - (float)start.tv_nsec)/m2);
		if (i > loops_reps/2){
			timetaken += time_usec;
		}
	}
	printf("Multiplication of two arrays:  %f\n",finalans);
	free(x);
	free(y);
	float mult1 = 1000000.0;
	float secondaveragehalftime = timetaken/((loops_reps - loops_reps/2)*mult1);
	float flopstime = (2.0*(float)length_of_array)/(secondaveragehalftime);
	float bandwidth = (length_of_array*sizeof(float)*2)/(secondaveragehalftime*mult1*1000);
	printf("N: %ld  <T>: %f sec  B: %f GB/sec  F: %f FLOP/sec\n",length_of_array,secondaveragehalftime,bandwidth,flopstime);
    return 0;
}