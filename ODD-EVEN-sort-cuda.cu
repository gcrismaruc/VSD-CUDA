%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


//Device functions can only be called from other device or global functions. __device__ functions cannot be called from host code.

//Global functions are also called "kernels". It's the functions that you may call from the host side using CUDA kernel call semantics (<<<...>>>).
__global__ void testKernel(long *in, long *out, long size){

	bool oddeven = true;
	//__shared__ for shared memory
	__shared__ bool swappedodd;
	__shared__ bool swappedeven;
	
	long temp;

	while(1){
	
		if(oddeven == true){
			

			__syncthreads();

			swappedeven=false;

			__syncthreads();

			//first column only which would have the array
			if (threadIdx.y == 0) {
			
				int idx = threadIdx.x;
				
				//0, 1, 2 threads will go through
				if( idx < (size/2) ){
					//COMPARISONS:
					// 0 <--> 1
					// 2 <--> 3
					// 4 <--> 5
					if ( in[2*idx] > in[2*idx+1] ){
						//BUBBLE SORT LOGIC
						temp= in[2*idx];
						in[2*idx]=in[2*idx+1];
						in[2*idx+1]=temp;
						swappedeven=true;
					
					}
				}
			}
			__syncthreads();
		}
		else{

			__syncthreads();

			swappedodd=false;

			__syncthreads();

			if (threadIdx.y == 0) {

				int idx = threadIdx.x;
				//0, 1 will go through
				if( idx < (size/2)-1 ){
					//COMPARISONS:
					// 1 <--> 2
					// 3 <--> 4
					if ( in[2*idx+1] > in[2*idx+2] ){

						temp=in[2*idx+1];
						in[2*idx+1]=in[2*idx+2];
						in[2*idx+2]=temp;
						swappedodd=true;

					}

				}


			}

			__syncthreads();

		}
	
	//if there are no swaps in odd phase as well as even phase then break (which means all sorting is done)
	// !(false) => true
	if( !( swappedodd || swappedeven ) )
		break;

	oddeven =! oddeven;	//switch phase of sorting

	}

	__syncthreads();

	//Store this phase's in[] array to out[] array
	int idx = threadIdx.x;

	if ( idx < size )
		out[idx] = in[idx];
		
}


int main(void)
{
	long i;
	long *a, *a_sorted;
	long *d_a, *d_sorted;
	//int n = 1* 1000 * 10;		//make sure to keep this even
	
    long n = 1000;
    long size = sizeof(long) *n;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);


	cudaMalloc( (void**) &d_a, size);
	cudaMalloc( (void**) &d_sorted, size);

	a = (long*) malloc(size);
	a_sorted = (long*) malloc(size);

	
   time_t t;
   
   
   /* Intializes random number generator */
   srand((unsigned) time(&t));
   int random_nr; 
	for(i = 0; i < n; i++) 
	{
      random_nr = rand() % 100;
		a[i] = random_nr;
           // printf(" a[%d] = %ld ", i, a[i]);

	}
	
	//d_a -> destination. a -> source.
	//Host to device array copy
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	
	//<<< >>> CUDA semantic
	testKernel<<<3 ,n>>>(d_a, d_sorted, n);

	//Device to Host array for final display (I/O)
	cudaMemcpy(a_sorted, d_sorted, size, cudaMemcpyDeviceToHost);
	
	for (i=0;i<n;i++){
		printf("%ld, ",a_sorted[i]);
	}
	
	printf("\n");
    
    cudaThreadSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time spent: %.5f\n", milliseconds);
    
    size_t free_byte ;

        size_t total_byte ;

        cudaMemGetInfo( &free_byte, &total_byte ) ;

        
        double free_db = (double)free_byte ;

        double total_db = (double)total_byte ;

        double used_db = total_db - free_db ;

        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
	
	//free memory allocated by malloc and cudamalloc
	free(a);
	free(a_sorted);
	cudaFree(d_sorted);
	cudaFree(d_a);
}
