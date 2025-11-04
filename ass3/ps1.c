#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int cmp_asc(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int cmp_desc(const void *a, const void *b) {
    return (*(int*)b - *(int*)a);
}

int main() {
    int n, num_threads;
    printf("Enter number of elements: ");
    scanf("%d", &n);
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    int *a = malloc(n * sizeof(int));
    int *b = malloc(n * sizeof(int));

    printf("Enter elements of array a:\n");
    for(int i=0; i<n; i++) scanf("%d", &a[i]);
    printf("Enter elements of array b:\n");
    for(int i=0; i<n; i++) scanf("%d", &b[i]);

    qsort(a, n, sizeof(int), cmp_asc);
    qsort(b, n, sizeof(int), cmp_desc);

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();
    long long result = 0;
    #pragma omp parallel for reduction(+:result)
    for(int i=0; i<n; i++) result += (long long)a[i] * b[i];
    double end = omp_get_wtime();

    printf("Result: %lld\n", result);
    printf("Time taken: %f seconds\n", end - start);
    printf("Threads used: %d\n", num_threads);

    free(a); free(b);
    return 0;
}
