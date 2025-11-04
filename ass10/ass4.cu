#include <stdio.h>

__global__ void kernel(){
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int global_id = gy * (gridDim.x * blockDim.x) + gx;
  printf("block=(%d,%d) thread=(%d,%d) global_id=%d: Hello world!\n",
         blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, global_id);
}

int main(){
  dim3 grid(2, 2);
  dim3 block(2, 1);
  kernel<<<grid, block>>>();
  cudaDeviceSynchronize();
  return 0;
}