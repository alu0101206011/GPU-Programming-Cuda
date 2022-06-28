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
__global__ void incHist(const int *A, int numElementsA, int *histogram, int numElementsHistogram) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < numElementsA) {
    int increment = A[i] % M;
    atomicAdd(&histogram[increment], 1);
  }
}


// Host main
int main(void) {
  // Vector length to be used, and compute its size
  const int numElementsA = N;
  size_t sizeA = numElementsA * sizeof(int);

  // Allocate the host input vector A
  int *h_A = (int*)malloc(sizeA);

  // Verify that allocations succeeded
  if (h_A == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Initialize the host input vector with [0, 1000000] random range
  time_t t;
  srand((unsigned) time(&t));
  for (int i = 0; i < numElementsA; i++) {
    h_A[i] = rand() % N;
  }
  printf("Vector element number: %d\n\n", numElementsA);
  //show_vector(h_A, 0, 10); Comprobamos que añade números aleatorios

  // Allocate the device input vector A
  int *d_A = NULL;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&d_A, sizeA));

  // Copy the host input vector A in host memory to the device input vector in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  CUDA_CHECK_RETURN(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));

  // Vector length to be used, and compute its size
  int numElementsHistogram = M;
  size_t sizeHistogram = numElementsHistogram * sizeof(int);

  // Allocate the host input vector histograma
  int *h_histogram = (int*)malloc(sizeHistogram);
  if (h_histogram == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
  }

  // Allocate the device input vector histogram
  int *d_histogram = NULL;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&d_histogram, sizeHistogram));

  // Initialize the host input vector
  for (int i = 0; i < M; i++) {
    h_histogram[i] = 0;
  }

  // Calculate the number of blocks needed
  int threadsPerBlock = HILOSPORBLOQUE;
  int blocksPerGrid = (numElementsA + threadsPerBlock - 1) / threadsPerBlock;  

  // Copy the host input vector histogram in host memory to the device input vector in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  CUDA_CHECK_RETURN(cudaMemcpy(d_histogram, h_histogram,  sizeHistogram, cudaMemcpyHostToDevice));

  // Launch the reduccion_paralela CUDA Kernel
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  incHist<<<blocksPerGrid, threadsPerBlock>>>(d_A, sizeA, d_histogram, numElementsHistogram);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  CUDA_CHECK_RETURN(cudaGetLastError());

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // Get back the result histogram
  printf("Copy histogram result from the CUDA device to the host memory\n");
  CUDA_CHECK_RETURN(cudaMemcpy(h_histogram, d_histogram, sizeHistogram, cudaMemcpyDeviceToHost));

  // Vector check
  printf("\nHistogram: ");
  show_vector(h_histogram, 0, M);
  int acc = 0;
  for (int i = 0; i < M; i++) {
    acc += h_histogram[i];
  }
  printf("Histogram total data: %d\n", acc);

  // Free device global memory
  CUDA_CHECK_RETURN(cudaFree(d_A));
  CUDA_CHECK_RETURN(cudaFree(d_histogram));

  // Free host memory
  free(h_A);
  free(h_histogram);

  printf("\nTiempo total: %f milisegundos\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("Done\n");
  return EXIT_SUCCESS;
}


// Check the return value of the CUDA runtime API call and exit the application if the call has failed.
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}


// Returns a range given the vector by the terminal
void show_vector(int* vector, int min, int max) {
  printf("[%d", vector[min]);
  for (unsigned i = min + 1; i < max; i++) 
    printf(", %d", vector[i]);   
  printf("]\n");
}