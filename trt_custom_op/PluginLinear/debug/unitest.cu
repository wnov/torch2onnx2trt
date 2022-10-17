#include <cuda_runtime.h>
#include <iostream>


#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

__global__ void matrixTranspose(const float *input, float *output, const int height, const int width){
    __shared__ float buffer[32][32 + 1];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width || y < height)
        buffer[y][x] = input[y * width + x];
    __syncthreads();

    if (y >= width || x >= height)
        return;
    
    output[y * height + x] = buffer[x][y];
}

void initMat(float *m, int height, int width){
    for (int i=0; i <height; i++){
        for (int j=0; j < width; j++){
            m[i * width + j] = i*10 + j;
        }
    }
}

void printMat(float *m, int height, int width){
    for (int i=0; i <height; i++){
        for (int j=0; j < width; j++){
            std::cout << m[i * width + j] << ", \t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(void){
    int height = 6, width = 8;
    float *m1 = (float *)malloc(sizeof(float) * height * width);
    float *m4 = (float *)malloc(sizeof(float) * height * width);
    initMat(m1, height, width);
    printMat(m1, height, width);

    float *m2 = nullptr ,*m3 = nullptr;
    CHECK(cudaMalloc((void**)&m3, sizeof(float) * height * width));
    CHECK(cudaMalloc((void**)&m2, sizeof(float) * height * width));

    CHECK(cudaMemcpy(m3, m1, sizeof(float)*height*width, cudaMemcpyHostToDevice));

    dim3 block(32, 32, 1), grid((height - 1)/32 + 1, (width - 1)/32 + 1, 1);
    matrixTranspose<<<grid, block>>>(m3, m2, height, width);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(m4, m2, sizeof(float1)*height*width, cudaMemcpyDeviceToHost));

    printMat(m4, width, height);
}