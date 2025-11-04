#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    long long num_steps;
    int num_threads;

    double step;
    double sum = 0.0;
    double start_time, run_time;

    printf("Enter number of intervals: ");
    scanf("%lld", &num_steps);
    
    printf("Enter number of threads: ");
    scanf("%d", &num_threads);

    step = 1.0 / (double)num_steps;

    omp_set_num_threads(num_threads);
    
    printf("\nPi Calculation\n");
    printf("Intervals: %lld\n", num_steps);
    printf("Threads: %d\n", num_threads);

    start_time = omp_get_wtime();
    
    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    double pi = step * sum;

    run_time = omp_get_wtime() - start_time;

    printf("Calculated Pi = %.15f\n", pi);
    printf("Execution Time: %f seconds\n", run_time);

    return 0;
}