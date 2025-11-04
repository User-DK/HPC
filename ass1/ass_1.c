#include <stdio.h>
#include <omp.h>

int main(void)
{
#pragma omp parallel
  {
    printf("Hello, world.\n");
    printf("Current Thread ID: %d\n", omp_get_thread_num());
  }
  return 0;
}