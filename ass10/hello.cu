#include <stdio.h>

__global__ void say_hello(){
  int i = threadIdx.x;
  printf("Hello, World from GPU thread idx: %d!\n", i);
}

int main(){
  // Launch kernel with 1 block and 1 thread
  say_hello<<<5, 5>>>();
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  return 0;
}