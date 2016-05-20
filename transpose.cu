#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;

#define TileDIM 32
#define BlockSize 8


__global__ 
void matTrans(dtype* AT, dtype* A, int N)  {
 
   __shared__ dtype temp[TileDIM+1][TileDIM+1];
    
  int Xindex = TileDIM * blockIdx.x + threadIdx.x;
  int Yindex = TileDIM * blockIdx.y + threadIdx.y;
  unsigned int i = 0;
	
  while(i < TileDIM){
	temp[threadIdx.y + i][threadIdx.x] = A[(i + Yindex) * gridDim.x * TileDIM + Xindex];
	i = i + BlockSize;
  }
  __syncthreads();

  Xindex = blockIdx.y * TileDIM + threadIdx.x;  
  Yindex = blockIdx.x * TileDIM + threadIdx.y;
  
  i = 0;
  while(i < TileDIM){
        AT[ (Yindex+i) * gridDim.x * TileDIM + Xindex] = temp[threadIdx.x][threadIdx.y + i];
	i = i + BlockSize;
  }
}

void
parseArg (int argc, char** argv, int* N)
{
	if(argc == 2) {
		*N = atoi (argv[1]);
		assert (*N > 0);
	} else {
		fprintf (stderr, "usage: %s <N>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
}


void
initArr (dtype* in, int N)
{
	int i;

	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void
cpuTranspose (dtype* A, dtype* AT, int N)
{
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			AT[j * N + i] = A[i * N + j];
		}
	}
}

int
cmpArr (dtype* a, dtype* b, int N)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
		if(abs(a[i] - b[i]) > 1e-6) cnt++;
	}

	return cnt;
}



void
gpuTranspose (dtype* A, dtype* AT, int N)
{
  struct stopwatch_t* timer = NULL;
  long double t_gpu;

	
  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

  stopwatch_start (timer);
	/* run your kernel here */

  dtype *d_in, *d_out;
  cudaMalloc(&d_in, N * N * sizeof (dtype));
  cudaMalloc(&d_out, N * N * sizeof (dtype));

  cudaMemset(d_in, 0.0,N * N * sizeof (dtype));
  cudaMemset(d_out, 0.0,N * N * sizeof (dtype));

  cudaMemcpy(d_in, A, N * N * sizeof (dtype), cudaMemcpyHostToDevice);

  int s = N;

  dim3 gb(TileDIM, BlockSize,1);
  dim3 tb(N/TileDIM,N/TileDIM,1);
  
  matTrans<<<tb,gb>>>(d_out,d_in,s);

  cudaThreadSynchronize ();

  cudaMemcpy(AT, d_out,  N * N * sizeof (dtype), cudaMemcpyDeviceToHost); 

  t_gpu = stopwatch_stop (timer);
  fprintf (stderr, "GPU transpose: %Lg secs ==> %Lg billion elements/second\n",
           t_gpu, (N * N) / t_gpu * 1e-9 );

}

int 
main(int argc, char** argv)
{
  /* variables */
	dtype *A, *ATgpu, *ATcpu;
  int err;

	int N;

  struct stopwatch_t* timer = NULL;
  long double t_cpu;


	N = -1;
	parseArg (argc, argv, &N);

  /* input and output matrices on host */
  /* output */
  ATcpu = (dtype*) malloc (N * N * sizeof (dtype));
  ATgpu = (dtype*) malloc (N * N * sizeof (dtype));

  /* input */
  A = (dtype*) malloc (N * N * sizeof (dtype));

	initArr (A, N * N);

	/* GPU transpose kernel */
	gpuTranspose (A, ATgpu, N);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

	stopwatch_start (timer);
  /* compute reference array */
	cpuTranspose (A, ATcpu, N);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU transpose kernel: %Lg secs\n",
           t_cpu);

  /* check correctness */
	err = cmpArr (ATgpu, ATcpu, N * N);
	if(err) {
		fprintf (stderr, "Transpose failed: %d\n", err);
	} else {
		fprintf (stderr, "Transpose successful\n");
	}
	
	free (A);
	free (ATgpu);
	free (ATcpu);

  return 0;
}
