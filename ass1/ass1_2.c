#include <stdio.h>
#include <omp.h> 

int main()
{
  int num_threads;

  printf("Please enter the number of threads to use: ");
  scanf("%d", &num_threads);

  printf("\n--- Running in Sequential ---\n");
  for (int i = 0; i < num_threads; ++i)
  {
    printf("Hello, World (Sequential Iteration %d)\n", i + 1);
  }

  printf("\n--- Running in Parallel ---\n");
  omp_set_num_threads(num_threads);
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();

    printf("Hello, World from thread %d\n", thread_id);
  }

  printf("\nExecution finished.\n");
  return 0;
}