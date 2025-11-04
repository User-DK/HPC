#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    long long n;
    int num_threads;

    double *a;
    double *c;
    double scalar = 3.14;
    double start_time, run_time;

    printf("Enter vector size: ");
    scanf("%lld", &n);
    
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    a = (double *)malloc(n * sizeof(double));
    c = (double *)malloc(n * sizeof(double));
    if (a == NULL || c == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    for (long long i = 0; i < n; i++) {
        a[i] = (double)i;
    }

    omp_set_num_threads(num_threads);

    printf("\nVector-Scalar Addition\n");
    printf("Vector Size: %lld\n", n);
    printf("Threads: %d\n", num_threads);

    start_time = omp_get_wtime();

    #pragma omp parallel for
    for (long long i = 0; i < n; i++) {
        c[i] = a[i] + scalar;
    }

    run_time = omp_get_wtime() - start_time;

    printf("Execution Time: %f seconds\n", run_time);

    free(a);
    free(c);

    return 0;
}