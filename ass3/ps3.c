#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void vector_add_static(int n, int scalar, int threads, int chunk) {
    int *A = malloc(n * sizeof(int));
    int *C = malloc(n * sizeof(int));
    for(int i=0;i<n;i++) A[i] = rand() % 100;

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static,chunk) num_threads(threads)
    for(int i=0;i<n;i++) {
        C[i] = A[i] + scalar;
    }
    double end = omp_get_wtime();
    printf("[Static] n=%d, threads=%d, chunk=%d, time=%.6f seconds\n", n, threads, chunk, end-start);

    free(A); free(C);
}

void vector_add_dynamic(int n, int scalar, int threads, int chunk) {
    int *A = malloc(n * sizeof(int));
    int *C = malloc(n * sizeof(int));
    for(int i=0;i<n;i++) A[i] = rand() % 100;

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic,chunk) num_threads(threads)
    for(int i=0;i<n;i++) {
        C[i] = A[i] + scalar;
    }
    double end = omp_get_wtime();
    printf("[Dynamic] n=%d, threads=%d, chunk=%d, time=%.6f seconds\n", n, threads, chunk, end-start);

    free(A); free(C);
}

void demonstrate_nowait(int n, int scalar, int threads) {
    int *A = malloc(n * sizeof(int));
    int *B = malloc(n * sizeof(int));
    int *C = malloc(n * sizeof(int));
    for(int i=0;i<n;i++) { A[i] = rand()%100; B[i] = rand()%100; }

    double start = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for nowait
        for(int i=0;i<n;i++) A[i] += scalar;

        #pragma omp for
        for(int i=0;i<n;i++) C[i] = A[i] + B[i];
    }
    double end = omp_get_wtime();
    printf("[Nowait] n=%d, threads=%d, time=%.6f seconds\n", n, threads, end-start);

    free(A); free(B); free(C);
}

int main() {
    int n, scalar, threads, chunk;
    printf("Enter vector size (n): ");
    scanf("%d", &n);
    printf("Enter scalar value: ");
    scanf("%d", &scalar);
    printf("Enter number of threads: ");
    scanf("%d", &threads);
    printf("Enter chunk size: ");
    scanf("%d", &chunk);

    printf("\n--- Results ---\n");
    vector_add_static(n, scalar, threads, chunk);
    vector_add_dynamic(n, scalar, threads, chunk);
    demonstrate_nowait(n, scalar, threads);

    return 0;
}
