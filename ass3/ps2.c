#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void matrix_add(int n, int threads) {
    int **A = malloc(n * sizeof(int*));
    int **B = malloc(n * sizeof(int*));
    int **C = malloc(n * sizeof(int*));
    for(int i=0;i<n;i++) {
        A[i] = malloc(n * sizeof(int));
        B[i] = malloc(n * sizeof(int));
        C[i] = malloc(n * sizeof(int));
    }

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }

    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(threads) collapse(2)
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            C[i][j] = A[i][j] + B[i][j];
    double end = omp_get_wtime();

    printf("\n--- Matrix Addition Result ---\n");
    printf("Matrix size   : %d x %d\n", n, n);
    printf("Threads used  : %d\n", threads);
    printf("Time taken    : %.6f seconds\n", end - start);
    printf("-----------------------------\n\n");

    for(int i=0;i<n;i++) { free(A[i]); free(B[i]); free(C[i]); }
    free(A); free(B); free(C);
}

int main() {
    int n, threads;
    char cont;
    srand(time(NULL));
    do {
        printf("Enter matrix size (n): ");
        scanf("%d", &n);
        printf("Enter number of threads: ");
        scanf("%d", &threads);

        matrix_add(n, threads);

        printf("Run again? (y/n): ");
        scanf(" %c", &cont);
    } while(cont == 'y' || cont == 'Y');
    return 0;
}
