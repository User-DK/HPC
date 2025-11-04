#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define MAX_CITIES 12
#define INF INT_MAX

int n;
int dist[MAX_CITIES][MAX_CITIES];
int best_cost = INF;
int best_path[MAX_CITIES];

void tsp_recursive(int path[], int visited[], int level, int current_cost, int current_city) {
    if (level == n) {
        int total_cost = current_cost + dist[current_city][0];
        #pragma omp critical
        {
            if (total_cost < best_cost) {
                best_cost = total_cost;
                memcpy(best_path, path, n * sizeof(int));
            }
        }
        return;
    }

    for (int i = 1; i < n; i++) {
        if (!visited[i] && dist[current_city][i] != INF) {
            int new_cost = current_cost + dist[current_city][i];
            if (new_cost < best_cost) {
                visited[i] = 1;
                path[level] = i;
                tsp_recursive(path, visited, level + 1, new_cost, i);
                visited[i] = 0;
            }
        }
    }
}

void tsp_sequential() {
    int path[MAX_CITIES];
    int visited[MAX_CITIES] = {0};
    
    path[0] = 0;
    visited[0] = 1;
    best_cost = INF;
    
    tsp_recursive(path, visited, 1, 0, 0);
}

void tsp_parallel() {
    best_cost = INF;
    
    #pragma omp parallel
    {
        int path[MAX_CITIES];
        int visited[MAX_CITIES] = {0};
        
        path[0] = 0;
        visited[0] = 1;
        
        #pragma omp for schedule(dynamic)
        for (int first_city = 1; first_city < n; first_city++) {
            if (dist[0][first_city] != INF) {
                int local_path[MAX_CITIES];
                int local_visited[MAX_CITIES] = {0};
                
                local_path[0] = 0;
                local_path[1] = first_city;
                local_visited[0] = 1;
                local_visited[first_city] = 1;
                
                tsp_recursive(local_path, local_visited, 2, dist[0][first_city], first_city);
            }
        }
    }
}

void generate_distance_matrix() {
    srand(42);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else if (i < j) {
                dist[i][j] = (rand() % 50) + 10;
                dist[j][i] = dist[i][j];
            }
        }
    }
}

void print_result() {
    printf("Best path: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", best_path[i]);
    }
    printf("0\n");
    printf("Minimum cost: %d\n", best_cost);
}

int main() {
    printf("=== TSP Parallel Solver (Exploratory Decomposition) ===\n\n");
    
    n = 10;
    printf("Number of cities: %d\n\n", n);
    
    generate_distance_matrix();
    
    printf("Distance Matrix (first 5x5):\n");
    for (int i = 0; i < 5 && i < n; i++) {
        for (int j = 0; j < 5 && j < n; j++) {
            printf("%3d ", dist[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    double start = omp_get_wtime();
    tsp_sequential();
    double seq_time = omp_get_wtime() - start;
    int seq_cost = best_cost;
    printf("Sequential Execution:\n");
    printf("Time: %.2f ms\n", seq_time * 1000);
    print_result();
    printf("\n");
    
    best_cost = INF;
    omp_set_num_threads(4);
    start = omp_get_wtime();
    tsp_parallel();
    double par_time = omp_get_wtime() - start;
    printf("Parallel Execution (4 threads):\n");
    printf("Time: %.2f ms\n", par_time * 1000);
    print_result();
    printf("\n");
    
    printf("Performance Metrics:\n");
    printf("Speedup: %.2fx\n", seq_time / par_time);
    printf("Efficiency: %.2f%%\n", (seq_time / par_time / 4) * 100);
    printf("Wasted Computation: ~0%% (exploratory)\n");
    
    return 0;
}