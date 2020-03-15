// Author : Sree Gowri Addepalli
// NETID : sga297

#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cudnn.h>


// C1 - Tiled Convolution in CUDA.

// Declaring constants.

/* Input tensor - height, width, channels.
   Filter height, width, channels. 
   Filter dimensions can be restricted to be odd.
   Tile Height and width   
   The number of Iterations.       */ 
   
#define H   1024     
#define W	1024
#define C	3
#define FH  3
#define FW	3
#define K	64
#define TH  32
#define TW  32
#define ITER 5


/* Create Image in Memory. */

void loadImageInMem(int h, int w,int c, double *it)
{
    for (int ki = 0; ki < c; ++ki) {
      for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
          it[ki*w*h + j*w + i]= ki * (i+j);
        }
      }
    }
}

static double TimeInSecs(struct timespec* timSec){
    double tinSec = (double)timSec->tv_sec;
	double tinNSec = (double)timSec->tv_nsec;
    return  tinSec + tinNSec/ 1000000000.0;
}



// Convolution Kernel  
	
__global__ void convolution2DKernel(int c, int k, int h, int w, int fh, int fw, double *itg, double *gpuf, double *otg)
   {
     // Getting the padded tile size.
     int paddedTileSize = (blockDim.x +fw-1) * (blockDim.y + fh -1);
	 
	 // Linear Indexing
	 int index = blockDim.x*threadIdx.y + threadIdx.x;
	 
	 // shared memory for padded image.
	 extern __shared__ double pImgX[];
	
    for (int o = index; o < paddedTileSize; o += (blockDim.x)*(blockDim.y)) {
    int x = (o % (blockDim.x+fw-1)) + (blockIdx.x * blockDim.x) - ((fw)/2);
    int y = (o / (blockDim.x+fw-1)) + (blockIdx.y * blockDim.y) - ((fh)/2);
	
    if (x < 0 || x >= w || y < 0 || y >= h) {
      for (int u = 0; u < c; u++)
        pImgX[o + u*paddedTileSize] = 0;
    } else {
      for (int u = 0; u < c; u++)
        pImgX[o + u*paddedTileSize] = itg[u*w*h + y*w + x];
    }

  }

  __syncthreads();
  
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;

  // calculation of convolution.
  for (int ki = 0; ki < k; ki++) {
    double output = 0;
    for (int ci = 0; ci < c; ci++) {
      for (int j = 0; j < fh; j++) {
        for (int i = 0; i < fw; i++) { 
          double filter = gpuf[ki*c*fh*fw + ci*fh*fw + (fh -1-j)*fw + (fw-1-i)];
          double input = pImgX[ci*paddedTileSize + (threadIdx.y+j)*(blockDim.x+fw-1) + threadIdx.x+i];
          output += input*filter;
        }
      }
    }

    if (x < w && y < h)
      otg[ki*h*w + y*w + x] = output;
  }
  
   
   }  

#define checkCUDNN(expression)                    \
{                                                  \
	cudnnStatus_t status = (expression);           			  \
	if (status != CUDNN_STATUS_SUCCESS) {        			    \
		printf("cuDNN error on line %d: %s\n" , __LINE__ ,	\
				cudnnGetErrorString(status));  						\
		exit(EXIT_FAILURE);                           \
	}                                               \
}


int main(int argc, char *argv[])
{
    // Input - Output Tensor for the image.
	
	double *it, *ot;
	double *itg, *otg;
	double ConvolutionTime = 0;
	double hostToDeviceTime =0;
	double hostToDeviceTimeCuDNN =0;
	double devToHostTime =0;
	double devToHostTimeCuDNN =0;
	double cuDNNtime = 0;
	double CompTime, TransferTime, TransferTimeDtH, TransferTimeh2D, TransferTimed2h;
	struct timespec startTime;
    struct timespec endTime;
	double *f;
	double *fT;
	double *fcuDNN;
	double *gpuf;
	double *gpufcuDNN;
	double filter[K][C][FH][FW];
	double filterCopy[K][C][FH][FW];
	
    size_t OPTensor = K*H*W*sizeof(double);
	size_t IPTensor = C*H*W*sizeof(double);
	
	
	// Allocating space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);
	
    
	// Allocating space in GPU. 
	cudaMalloc(&itg, IPTensor);
    cudaMalloc(&otg, OPTensor);
	
	// Tile Width and Height.
	dim3 tileSize(TW, TH);
	dim3 numOfTiles((((W)+(TW)-1)/(TW)), (((H)+(TH)-1)/(TH)));
	

    
	// Steps for creating a filter of size K*C*FW*FH
	cudaMalloc(&gpuf, (K*C*FW*FH)*sizeof(double));
	cudaMalloc(&gpufcuDNN, (K*C*FW*FH)*sizeof(double));
	f = (double *)malloc((K*C*FW*FH)*sizeof(double));
	fT = (double *)malloc((K*C*FW*FH)*sizeof(double));
	fcuDNN = (double *)malloc((K*C*FW*FH)*sizeof(double));
	
	// Filling the input filter. (2 ways of doing it)
	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int j = 0; j < FH; j++) {
				for (int i = 0; i < FW; i++){
				     filter[k][c][j][i] = (c+k)*(i+j);
					 fT[k*C*FH*FW + c*FH*FW + j*FW + i] = (c+k)*(i+j);
					 }
			}
		}
	}
	//  Transposition.
	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int j = 0; j < FH; j++) {
				for (int i = 0; i < FW; i++)
				     filterCopy[k][c][j][i] = filter[k][c][FH-1-j][FW-1-i];
			}
		}
	}
	
	// Row order format.
	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int j = 0; j < FH; j++) {
				for (int i = 0; i < FW; i++)
					f[k*C*FH*FW + c*FH*FW + j*FW + i] = filterCopy[k][c][j][i];
			}
		}
	}
	
	// Row order format for filter for cuDNN.
	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int j = 0; j < FH; j++) {
				for (int i = 0; i < FW; i++)
					fcuDNN[k*C*FH*FW + c*FH*FW + j*FW + i] = filter[k][c][j][i];
			}
		}
	}
	
	// printing filter checksum.
	double fcs=0;
    for (int i = 0; i < (K*C*FH*FW); i++){
	fcs += fcuDNN[i];
	}
    //printf("filter tensor checksum for: %.2f\n",fcs);
	
	// Copying the filter cpu memory  to gpu.
	
	cudaMemcpy(gpuf, fT, (K*C*FH*FW)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpufcuDNN, fcuDNN, (K*C*FH*FW)*sizeof(double), cudaMemcpyHostToDevice);
    free(fT);
	free(fcuDNN);

	
	
	// Shared memory size for the tile.
	int sharedMemorySize = C*(TW+FW-1)*(TH+FH-1)*sizeof(double);
	
	printf("\n Convolution with CUDA:\n");
	for(int i=0; i<ITER; i++){
	printf("\n\n");
	printf("Iteration number: %d\n", i+1);
	  
	// Allocating space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);
    
	// Allocating space in GPU. 
	cudaMalloc(&itg, IPTensor);
    cudaMalloc(&otg, OPTensor);
	
	// Row major ordering.
	loadImageInMem(H, W, C, it);
	
	// checking he checksum for input tensor.
	double ics=0;
    for (int i = 0; i < (H*C*W); i++){
	ics += it[i];
	}
    printf("cuda input tensor checksum: %.2f\n",ics);
	
	// Time to copy from CPU to GPU.
	clock_gettime(CLOCK_MONOTONIC, &startTime);
    cudaMemcpy(itg, it, IPTensor, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    TransferTime = TimeInSecs(&endTime) - TimeInSecs(&startTime);
	hostToDeviceTime += TransferTime;
    printf("Copy host->dev kernel %lf sec\n", TransferTime);
	
	
	// Time to kernel launch.
	clock_gettime(CLOCK_MONOTONIC, &startTime);
    //Function to call for kernel launch
	convolution2DKernel<<<numOfTiles, tileSize, sharedMemorySize>>>(C, K, H, W, FH, FW, itg, gpuf, otg);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    CompTime = TimeInSecs(&endTime) - TimeInSecs(&startTime);
	ConvolutionTime += CompTime;
    printf("Time kernel %lf sec\n", CompTime);
	
	
    // Time to copy from GPU to CPU.
	clock_gettime(CLOCK_MONOTONIC, &startTime);
    cudaMemcpy(ot, otg, OPTensor, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    TransferTimeDtH = TimeInSecs(&endTime) - TimeInSecs(&startTime);
	devToHostTime += TransferTimeDtH; 
    printf("Copy dev->host kernel %lf sec\n", TransferTimeDtH);
	
	// output tensor checksum
	double ocs=0;
    for (int i = 0; i < (H*K*W); i++){
	ocs += ot[i];
	}
    printf("Cuda output tensor checksum: %.2f\n",ocs);  
	}
	
	ConvolutionTime = ConvolutionTime/ITER;
	hostToDeviceTime = hostToDeviceTime/ITER;
	devToHostTime = devToHostTime/ITER;
	
	
	// C2 - Convolution with cuDNN.
	// Reference: http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
	
	cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
	
	cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_DOUBLE,
                                      /*batch_size=*/1,
                                      /*channels=*/C,
                                      /*image_height=*/H,
                                      /*image_width=*/W));
									  
	
	cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*dataType=*/CUDNN_DATA_DOUBLE,
                                      /*batch_size=*/1,
                                      /*channels=*/K,
                                      /*image_height=*/H,
                                      /*image_width=*/W));
									  
	
	cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNN_DATA_DOUBLE,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/K,
                                      /*in_channels=*/C,
                                      /*kernel_height=*/FH,
                                      /*kernel_width=*/FW));
									  
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CONVOLUTION,
                                           /*computeType=*/CUDNN_DATA_DOUBLE));
									  
	cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));

    
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
    
	
	void* d_workspace;
    cudaMalloc(&d_workspace, workspace_bytes);

    double alpha = 1, beta = 0.0;

    printf("\n Convolution with CUDNN.\n");
	for(int i=0; i<ITER; i++){
	printf("\n\n");
	printf("Iteration number: %d\n", i+1);
	  
	// Allocating space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);
    
	// Allocating space in GPU. 
	cudaMalloc(&itg, IPTensor);
    cudaMalloc(&otg, OPTensor);
	
	// Row major ordering.
	loadImageInMem(H, W, C, it);
	
	// checking he checksum for input tensor.
	double ics=0;
    for (int i = 0; i < (H*C*W); i++){
	ics += it[i];
	}
    printf("cuda input tensor checksum: %.2f\n",ics);
	
	// Time to copy from CPU to GPU.
	
	clock_gettime(CLOCK_MONOTONIC, &startTime);
    cudaMemcpy(itg, it, IPTensor, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    TransferTimeh2D = TimeInSecs(&endTime) - TimeInSecs(&startTime);
	hostToDeviceTimeCuDNN += TransferTimeh2D;
    printf("Copy host->dev kernel %lf sec\n", TransferTimeh2D);
	
	
	// Time to Run CuDNN Kernel.
	
	clock_gettime(CLOCK_MONOTONIC, &startTime);
    //Function to call for kernel launch
	//convolution2DKernel<<<numOfTiles, tileSize, sharedMemorySize>>>(C, K, H, W, FH, FW, itg, gpufcuDNN, otg);
	
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alpha,
                                   input_descriptor,
                                   itg, 
                                   kernel_descriptor,
                                   gpufcuDNN, 
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace, 
                                   workspace_bytes,
                                   &beta,
                                   output_descriptor,
                                   otg)); 
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    CompTime = TimeInSecs(&endTime) - TimeInSecs(&startTime);
	cuDNNtime += CompTime;
    printf("Time kernel %lf sec\n", CompTime);
	
	
    // Time to copy from GPU to CPU.
	clock_gettime(CLOCK_MONOTONIC, &startTime);
    cudaMemcpy(ot, otg, OPTensor, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    TransferTimed2h = TimeInSecs(&endTime) - TimeInSecs(&startTime);
	devToHostTimeCuDNN += TransferTimed2h;
    printf("Copy dev->host kernel %lf sec\n", TransferTimed2h);
	
	// Output tensor checksum.
	
	double ocs=0;
    for (int i = 0; i < (H*K*W); i++){
	ocs += ot[i];
	}
    printf("cuda output tensor checksum: %.2f\n",ocs);  
	}
	
	cuDNNtime = cuDNNtime/ITER;
	hostToDeviceTimeCuDNN = hostToDeviceTimeCuDNN/ITER;
	devToHostTimeCuDNN = devToHostTimeCuDNN/ITER;
	
	printf("\n\n <Average Convolution Time>: Conv %lf sec  cuDNN %lf sec\n", ConvolutionTime, cuDNNtime);
	printf("\n\n <Average Host to Device Time>: Conv %lf sec  cuDNN %lf sec\n", hostToDeviceTime, hostToDeviceTimeCuDNN);
	printf("\n\n <Average Device to Host Time>: Conv %lf sec  cuDNN %lf sec\n", devToHostTime, devToHostTimeCuDNN);
	
	return 0;	
}	








