/*
 ============================================================================
 Proyecto GPU
 Realizado por: Anabel Díaz Labrador

 ============================================================================
 */


#include <iostream>

#include <cuda_runtime.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

void show_vector(int*, int, int);

#define N (1048576)
#define M (8)
#define HILOSPORBLOQUE (512)


// Device kernel
__global__ void incHist(const int *A, int numElements, int *histogram, int numElementsHistograms) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < numElements) {
    int increment = A[i] % M;
    atomicAdd(&histogram[M * blockIdx.x + increment], 1);
  }
}

__global__ void reduccion_paralela(int *histogram, int numElements, int *result) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    int middle = numElements / 2;
    while (middle >= M) {  // Hacemos reducción hasta que queden por juntar 8. Ultima iteración middle = 4
      if (i < middle) {
        histogram[i] = histogram[i] + histogram[i + middle];
      }
      __syncthreads();
      middle = middle / 2;
    } 
  } 

  if (i >= 0 && i < M) {
    result[i] = histogram[i];
  }
}


// Host main
int main(void) {
  // Vector length to be used, and compute its size
  const int numElementsA = N;
  size_t sizeA = numElementsA * sizeof(int);

  // Allocate Unified Memory -- accessible from CPU or GPU
  int *A;
  CUDA_CHECK_RETURN(cudaMallocManaged(&A, sizeA));
  
  // Verify that allocations succeeded
  if (A == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Initialize the host input vector with [0, 1000000] random range
  time_t t;
  srand((unsigned) time(&t));
  for (int i = 0; i < numElementsA; i++) {
    A[i] = rand() % N;
  }
  printf("Vector element number: %d\n", numElementsA);

  // Calculamos el número de bloques necesario
  int threadsPerBlock = HILOSPORBLOQUE;
  int blocksPerGrid = (numElementsA + threadsPerBlock - 1) / threadsPerBlock;

  // Vector length to be used, and compute its size
  int numElementsHistograms = blocksPerGrid * M;
  size_t sizeHistograms = numElementsHistograms * sizeof(int);

  // Allocate Unified Memory -- accessible from CPU or GPU
  int *histograms;
  CUDA_CHECK_RETURN(cudaMallocManaged(&histograms, sizeHistograms));

  if (histograms == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Initialize the host input vector
  for (int i = 0; i < numElementsHistograms; i++) {
    histograms[i] = 0;
  }

  // Launch the incHist CUDA Kernel
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  incHist<<<blocksPerGrid, threadsPerBlock>>>(A, numElementsA, histograms, numElementsHistograms);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaGetLastError());

  float elapsedTime1;
  cudaEventElapsedTime(&elapsedTime1, start, stop);
  
  // Checkeamos vector
  printf("\nHistogram: ");
  show_vector(histograms, 0, M);
  int acc = 0;
  for (int i = 0; i < numElementsHistograms; i++) {
    acc += histograms[i];
  }
  printf("Histogram total increments: %d\nHistogram size: %d\n", acc, numElementsHistograms);

  // Vector length to be used, and compute its size
  int numElementsHistogram = M;
  size_t sizeHistogram = numElementsHistogram * sizeof(int);

  // Allocate Unified Memory -- accessible from CPU or GPU
  int *histogram;
  CUDA_CHECK_RETURN(cudaMallocManaged(&histogram, sizeHistogram));
  if (histogram == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Initialize the host input vector
  for (int i = 0; i < M; i++) {
    histogram[i] = 0;
  }

  // Launch the reduccion_paralela CUDA Kernel
  blocksPerGrid = (numElementsHistograms + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  cudaEventRecord(start, 0);
  reduccion_paralela<<<blocksPerGrid, threadsPerBlock>>>(histograms, numElementsHistograms, histogram);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaGetLastError());

  float elapsedTime2;
  cudaEventElapsedTime(&elapsedTime2, start, stop);

  // Checkeamos vector
  show_vector(histogram, 0, M);
  acc = 0;
  for (int i = 0; i < M; i++) {
    acc += histogram[i];
  }
  printf("Histogram total data: %d\n", acc);

  // Free device global memory
  CUDA_CHECK_RETURN(cudaFree(A));
  CUDA_CHECK_RETURN(cudaFree(histograms));
  CUDA_CHECK_RETURN(cudaFree(histogram));

  printf("\nTiempo construyendo histogramas locales: %f milisegundos\n", elapsedTime1);
  printf("Tiempo juntando histogramas en uno final: %f milisegundos\n", elapsedTime2);
  printf("Tiempo total: %f milisegundos\n", elapsedTime1 + elapsedTime2);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Done\n");
  return EXIT_SUCCESS;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}


void show_vector(int* vector, int min, int max) {
  printf("[%d", vector[min]);
  for (unsigned i = min + 1; i < max; i++) 
    printf(", %d", vector[i]);   
  printf("]\n");
}