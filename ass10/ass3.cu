#include <stdio.h>

__global__ void kernel(){
  printf("Hello from block %d\n Hello world!\n", blockIdx.x);
}

int main(){
  kernel<<<5, 1>>>();
  return 0;
}