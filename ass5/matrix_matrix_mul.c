#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int n, num_threads;
    printf("Enter matrix size (n): ");
    scanf("%d", &n);
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    omp_set_num_threads(num_threads);

    double **A = malloc(n * sizeof(double*));
    double **B = malloc(n * sizeof(double*));
    double **C = malloc(n * sizeof(double*));

    for(int i=0;i<n;i++) {
        A[i] = malloc(n * sizeof(double));
        B[i] = malloc(n * sizeof(double));
        C[i] = malloc(n * sizeof(double));
    }

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++) {
            A[i][j] = rand()%100;
            B[i][j] = rand()%100;
            C[i][j] = 0.0;
        }

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                C[i][j] += A[i][k] * B[k][j];
    double end = omp_get_wtime();

    printf("\n===== Matrix Multiplication Results =====\n");
    printf("Matrix size      : %d x %d\n", n, n);
    printf("Threads used     : %d\n", num_threads);
    printf("Time taken       : %.6f seconds\n", end-start);
    printf("========================================\n");

    for(int i=0;i<n;i++) { free(A[i]); free(B[i]); free(C[i]); }
    free(A); free(B); free(C);
    return 0;
}
