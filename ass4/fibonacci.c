#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

long long fib(int n) {
    if(n<=1) return n;
    long long x, y;
    #pragma omp task shared(x)
    x = fib(n-1);
    #pragma omp task shared(y)
    y = fib(n-2);
    #pragma omp taskwait
    return x + y;
}

int main() {
    int n, threads;
    printf("Enter Fibonacci number to compute (n): ");
    scanf("%d", &n);
    printf("Enter number of threads: ");
    scanf("%d", &threads);

    omp_set_num_threads(threads);

    long long result;
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        result = fib(n);
    }
    double end = omp_get_wtime();

    printf("\n--- Fibonacci Computation ---\n");
    printf("Input n           : %d\n", n);
    printf("Threads used      : %d\n", threads);
    printf("Fib(%d)           : %lld\n", n, result);
    printf("Time taken (secs) : %.6f\n", end - start);

    return 0;
}
