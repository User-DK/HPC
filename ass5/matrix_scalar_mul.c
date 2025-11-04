#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n, threads;
    double scalar;

    printf("Enter matrix size (n): ");
    scanf("%d", &n);
    printf("Enter scalar value: ");
    scanf("%lf", &scalar);
    printf("Enter number of threads: ");
    scanf("%d", &threads);

    omp_set_num_threads(threads);

    double **A = malloc(n * sizeof(double*));
    double **C = malloc(n * sizeof(double*));
    for(int i=0;i<n;i++) {
        A[i] = malloc(n * sizeof(double));
        C[i] = malloc(n * sizeof(double));
    }

    // Initialize matrix A
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            A[i][j] = rand() % 100;

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            C[i][j] = scalar * A[i][j];
    double end = omp_get_wtime();

    printf("\n--- Matrix Scalar Multiplication ---\n");
    printf("Matrix size      : %d x %d\n", n, n);
    printf("Scalar value     : %.2f\n", scalar);
    printf("Threads used     : %d\n", threads);
    printf("Time taken       : %.6f seconds\n", end-start);

    for(int i=0;i<n;i++) { free(A[i]); free(C[i]); }
    free(A); free(C);
    return 0;
}
