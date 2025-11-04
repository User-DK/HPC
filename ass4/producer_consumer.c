#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int *buffer;
int in = 0, out = 0, count = 0;

int main() {
    int SIZE, ITEMS, THREADS;

    printf("Enter buffer size: ");
    scanf("%d", &SIZE);
    printf("Enter number of items: ");
    scanf("%d", &ITEMS);
    printf("Enter number of threads (>=2): ");
    scanf("%d", &THREADS);

    if (THREADS < 2) THREADS = 2;

    buffer = (int*)malloc(SIZE * sizeof(int));
    if (!buffer) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    omp_set_num_threads(THREADS);

    double start = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for(int i=1;i<=ITEMS;i++) {
                int produced = i;
                int placed = 0;
                while(!placed) {
                    #pragma omp
                    {
                        if(count < SIZE) {
                            buffer[in] = produced;
                            in = (in+1)%SIZE;
                            count++;
                            printf("Produced: %d\n", produced);
                            placed = 1;
                        }
                    }
                }
            }
        }

        #pragma omp section
        {
            for(int i=1;i<=ITEMS;i++) {
                int consumed;
                int taken = 0;
                while(!taken) {
                    #pragma omp critical
                    {
                        if(count > 0) {
                            consumed = buffer[out];
                            out = (out+1)%SIZE;
                            count--;
                            printf("Consumed: %d\n", consumed);
                            taken = 1;
                        }
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    printf("Time taken: %.6f seconds\n", end - start);

    free(buffer);
    return 0;
}
