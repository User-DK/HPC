#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n, num_threads;
    printf("Enter array size (n): ");
    scanf("%d", &n);
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    int *A = malloc(n * sizeof(int));
    int *P = malloc(n * sizeof(int));
    for(int i=0; i<n; i++) A[i] = i+1;

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();

    int *offsets = calloc(num_threads, sizeof(int));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk = (n + num_threads - 1) / num_threads;
        int start_idx = tid * chunk;
        int end_idx = (start_idx + chunk > n) ? n : start_idx + chunk;

        int local_sum = 0;
        for(int i=start_idx; i<end_idx; i++) {
            local_sum += A[i];
            P[i] = local_sum;
        }
        offsets[tid] = local_sum;

        #pragma omp barrier
        int add = 0;
        for(int i=0; i<tid; i++) add += offsets[i];

        for(int i=start_idx; i<end_idx; i++) {
            P[i] += add;
        }
    }

    double end = omp_get_wtime();

    printf("\nInput array size: %d\n", n);
    printf("Threads used: %d\n", num_threads);
    printf("Prefix sum: ");
    for(int i=0; i<n; i++) printf("%d ", P[i]);
    printf("\nTime taken: %.6f seconds\n", end-start);

    free(A); free(P); free(offsets);
    return 0;
}
