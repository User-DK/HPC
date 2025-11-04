#include <stdio.h>

__global__ void kernel(){
  printf("Hello from thread %d\n Hello world!\n", threadIdx.x);
}

int main(){
  kernel<<<1, 5>>>();
  return 0;
}