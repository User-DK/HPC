#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n, num_threads;
    printf("Enter matrix/vector size (n): ");
    scanf("%d", &n);
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    double **A = malloc(n * sizeof(double*));
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));

    for(int i=0;i<n;i++) {
        A[i] = malloc(n * sizeof(double));
    }

    // Initialize matrix and vector
    for(int i=0;i<n;i++) {
        x[i] = rand()%100;
        y[i] = 0.0;
        for(int j=0;j<n;j++)
            A[i][j] = rand()%100;
    }

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;i++) {
        double sum = 0.0;
        for(int j=0;j<n;j++)
            sum += A[i][j] * x[j];
        y[i] = sum;
    }
    double end = omp_get_wtime();

    printf("\n===== Matrix-Vector Multiplication =====\n");
    printf("Matrix size (n): %d\n", n);
    printf("Threads used: %d\n", num_threads);
    printf("Time taken: %f seconds\n", end-start);
    printf("========================================\n");

    for(int i=0;i<n;i++) free(A[i]);
    free(A); free(x); free(y);
    return 0;
}
