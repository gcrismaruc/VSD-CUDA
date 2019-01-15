 %%cu
 # include <stdio.h>
 # include <stdlib.h>
 # include <cuda.h>

long n = 33000;

__global__ void testKernel(long *in, long *out, long size, long n) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	long temp;
    
	__shared__ bool swappedodd;
	__shared__ bool swappedeven;
    long i;
    
//    printf("Thread number: %d ;", index);
	for(i=0; i < n; i++) {
			if(n % 2 == 0) {
      if (i % 2 == 0) {
				__syncthreads();
				swappedeven = false;
				__syncthreads();

				if (index < (size / 2)) {
            //printf("Odd: Thread number: %d ;", index);
					if (in[index * 2] > in[index * 2 + 1]) {
              //printf("Odd: Thread %d change: %ld <-> %ld ;", index, in[index*2], in[index*2 +1]);
						temp = in[index * 2];
						in[index * 2] = in[index * 2 + 1];
						in[index * 2 + 1] = temp;
              swappedeven=true;
					}
				}
				__syncthreads();
			} else {
				__syncthreads();
				swappedodd = false;
				__syncthreads();
				if (index < (size / 2) - 1) {
            //printf("Even: Thread number: %d ;", index);
					if (in[index * 2 + 1] > in[index * 2 + 2]) {
                            //printf("Even: Thread %d change: %ld <-> %ld ;", index, in[index*2+1], in[index*2 +2]);

						temp = in[index * 2 + 1];
						in[index * 2 + 1] = in[index * 2 + 2];
						in[index * 2 + 2] = temp;
              swappedodd=true;
					}
				}
				__syncthreads();
			}
      } else
      {
          if (i % 2 == 0) {
				__syncthreads();
				swappedeven = false;
				__syncthreads();

				if (index < (size / 2)) {
            //printf("Odd: Thread number: %d ;", index);
					if (in[index * 2] > in[index * 2 + 1]) {
              //printf("Odd: Thread %d change: %ld <-> %ld ;", index, in[index*2], in[index*2 +1]);
						temp = in[index * 2];
						in[index * 2] = in[index * 2 + 1];
						in[index * 2 + 1] = temp;
              swappedeven=true;
					}
				}
				__syncthreads();
			} else {
				__syncthreads();
				swappedodd = false;
				__syncthreads();
				if (index < (size / 2)) {
            //printf("Even: Thread number: %d ;", index);
					if (in[index * 2 + 1] > in[index * 2 + 2]) {
                            //printf("Even: Thread %d change: %ld <-> %ld ;", index, in[index*2+1], in[index*2 +2]);

						temp = in[index * 2 + 1];
						in[index * 2 + 1] = in[index * 2 + 2];
						in[index * 2 + 2] = temp;
              swappedodd=true;
					}
				}
				__syncthreads();
			}
      }
		
      if (!(swappedodd || swappedeven))
		      break;

	   }

    __syncthreads();
    //printf("Final: Thread %d val = %ld ;", index, in[index]);
     //   printf("Final: Thread %d val = %ld ;", size / 2 + index, in[size / 2 + index]);

    out[index] = in[index];
    out[size / 2 + index] = in[size / 2 + index];
    
    if(n % 2 != 0) {
        if(index == size / 2 - 1) {
            out[size - 1] = in[size - 1];
        }
    }
}

int main(void) {
	long i;
	long * a,
	 * a_sorted;
	long * d_a,
	 * d_sorted;
	//int n = 1* 1000 * 10;		//make sure to keep this even


	long size = sizeof(long) * n;
	cudaEvent_t start,
	stop;

	cudaEventCreate( & start);
	cudaEventCreate( & stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void ** ) & d_a, size);
	cudaMalloc((void ** ) & d_sorted, size);

	a = (long * )malloc(size);
	a_sorted = (long * )malloc(size);

	time_t t;

	/* Intializes random number generator */
	srand((unsigned)time( & t));
	int random_nr;
	for (i = 0; i < n; i++) {
		random_nr = rand() % 100;
		a[i] = random_nr;
		//printf(" a[%d] = %ld ", i, a[i]);
    printf("%ld ;", a[i]);
	}

	//d_a -> destination. a -> source.
	//Host to device array copy
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	//<<< >>> CUDA semantic
    int nr_blocks;
    int nr_threads;
    
    if(n > 512) {
        nr_blocks = n / 512 + 1;
        nr_threads = 512;
    } else {
        nr_blocks = 1;
        nr_threads = n;
    }
	testKernel <<< nr_blocks,	nr_threads>>> (d_a, d_sorted, n, n);

	//Device to Host array for final display (I/O)
	cudaMemcpy(a_sorted, d_sorted, size, cudaMemcpyDeviceToHost);

    printf("Sorted: ");
	for (i = 0; i < n; i++) {
		printf("%ld, ", a_sorted[i]);
	}

	printf("\n");

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime( & milliseconds, start, stop);
	printf("Time spent: %.5f\n", milliseconds);

	size_t free_byte;

	size_t total_byte;

	cudaMemGetInfo( & free_byte,  & total_byte);

	double free_db = (double)free_byte;

	double total_db = (double)total_byte;

	double used_db = total_db - free_db;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//free memory allocated by malloc and cudamalloc
	free(a);
	free(a_sorted);
	cudaFree(d_sorted);
	cudaFree(d_a);
}
