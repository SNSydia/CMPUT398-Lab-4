#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <wb.h>

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
	int numAColumns, int numBRows, int numBColumns) {
	// TODO: Insert code to implement matrix multiplication here

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int numCRows = numARows;
	int numCColumns = numBColumns;


	float sum = 0.0;

	if (row < numARows && col < numBColumns)
	{
		for (int k = 0; k < numBRows; k++)
		{
			sum += A[row * numAColumns + k] * B[k * numBColumns + col];
			C[row * numCColumns + col] = sum;
		}
		
		//C[row * numCColumns + col] = sum;
	}

}

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
	    }                                                                     \
    } while (0)

int main(int argc, char **argv) {
	wbArg_t args;
	float *hostA; // The A matrix
	float *hostB; // The B matrix
	float *hostC; // The output C matrix
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows;    // number of rows in the matrix A
	int numAColumns; // number of columns in the matrix A
	int numBRows;    // number of rows in the matrix B
	int numBColumns; // number of columns in the matrix B
	int numCRows;
	int numCColumns;

	args = wbArg_read(argc, argv);

#if LAB_DEBUG
	std::cout << "Running GPU Matrix Multiplicaion ..." << std::endl;
#endif

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
		&numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
		&numBColumns);
	// TODO: Allocate the hostC matrix
	hostC = (float *)malloc(numARows * numBColumns * sizeof(float));
	
	wbTime_stop(Generic, "Importing data and creating memory on host");

	// TODO: Set numCRows and numCColumns
	numCRows = numARows;
	numCColumns = numBColumns;

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	// TODO: Allocate GPU memory here
	cudaMalloc((void **) &deviceA, (unsigned long long) (numARows * numAColumns * sizeof(float)));
	cudaMalloc((void **) &deviceB, (unsigned long long) (numBRows * numBColumns * sizeof(float)));
	cudaMalloc((void **) &deviceC, (unsigned long long) (numCRows * numCColumns * sizeof(float)));

	
	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	// TODO: Copy memory to the GPU here
	cudaMemcpy((void *) deviceA, (const void *) hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy((void *) deviceB, (const void *) hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);


	wbTime_stop(GPU, "Copying input memory to the GPU.");

	// TODO: Initialize the grid and block dimensions here
	// Here you will have to use dim3
	// dim3 blockDim( ... )
	// dim3 gridDim( ... )

	dim3 blockDim(32, 32);
	dim3 gridDim(32, 32);


	// wbLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);
	// wbLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);

	wbTime_start(Compute, "Performing CUDA computation");
	
	matrixMultiply << <gridDim, blockDim >> >(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

	cudaDeviceSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	// TODO:: Copy the GPU memory back to the CPU here
	cudaMemcpy((void *) hostC, (const void *) deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
	
	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	// TODO:: Free the GPU memory
	cudaFree((void *) deviceA);
	cudaFree((void *) deviceB);
	cudaFree((void *) deviceC);



	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostC, numCRows, numCColumns);

	free(hostA);
	free(hostB);
	free(hostC);

#if LAB_DEBUG
	system("pause");
#endif

	return 0;
}
