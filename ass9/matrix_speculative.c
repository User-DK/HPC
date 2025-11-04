#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#define N 512

double A[N][N], B[N][N];
double C_classical[N][N], C_strassen[N][N];
int classical_done = 0, strassen_done = 0;
double classical_time = 0, strassen_time = 0;

void matrix_init() {
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(rand() % 100) / 10.0;
            B[i][j] = (double)(rand() % 100) / 10.0;
        }
    }
}

void classical_multiply() {
    double start = omp_get_wtime();
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_classical[i][j] = 0;
            for (int k = 0; k < N; k++) {
                C_classical[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    classical_time = omp_get_wtime() - start;
    classical_done = 1;
}

void add_matrix(double X[N][N], double Y[N][N], double Z[N][N], int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            Z[i][j] = X[i][j] + Y[i][j];
}

void sub_matrix(double X[N][N], double Y[N][N], double Z[N][N], int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            Z[i][j] = X[i][j] - Y[i][j];
}

void strassen_multiply_simple() {
    double start = omp_get_wtime();
    
    int half = N / 2;
    double A11[N][N], A12[N][N], A21[N][N], A22[N][N];
    double B11[N][N], B12[N][N], B21[N][N], B22[N][N];
    double M1[N][N], M2[N][N], M3[N][N], M4[N][N], M5[N][N], M6[N][N], M7[N][N];
    double T1[N][N], T2[N][N];
    
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];
            
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];
            B22[i][j] = B[i + half][j + half];
        }
    }
    
    add_matrix(A11, A22, T1, half);
    add_matrix(B11, B22, T2, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M1[i][j] = 0;
            for (int k = 0; k < half; k++)
                M1[i][j] += T1[i][k] * T2[k][j];
        }
    
    add_matrix(A21, A22, T1, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M2[i][j] = 0;
            for (int k = 0; k < half; k++)
                M2[i][j] += T1[i][k] * B11[k][j];
        }
    
    sub_matrix(B12, B22, T2, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M3[i][j] = 0;
            for (int k = 0; k < half; k++)
                M3[i][j] += A11[i][k] * T2[k][j];
        }
    
    sub_matrix(B21, B11, T2, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M4[i][j] = 0;
            for (int k = 0; k < half; k++)
                M4[i][j] += A22[i][k] * T2[k][j];
        }
    
    add_matrix(A11, A12, T1, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M5[i][j] = 0;
            for (int k = 0; k < half; k++)
                M5[i][j] += T1[i][k] * B22[k][j];
        }
    
    sub_matrix(A21, A11, T1, half);
    add_matrix(B11, B12, T2, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M6[i][j] = 0;
            for (int k = 0; k < half; k++)
                M6[i][j] += T1[i][k] * T2[k][j];
        }
    
    sub_matrix(A12, A22, T1, half);
    add_matrix(B21, B22, T2, half);
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            M7[i][j] = 0;
            for (int k = 0; k < half; k++)
                M7[i][j] += T1[i][k] * T2[k][j];
        }
    
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C_strassen[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C_strassen[i][j + half] = M3[i][j] + M5[i][j];
            C_strassen[i + half][j] = M2[i][j] + M4[i][j];
            C_strassen[i + half][j + half] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }
    }
    
    strassen_time = omp_get_wtime() - start;
    strassen_done = 1;
}

void sequential_execution() {
    printf("=== Sequential Execution ===\n");
    
    classical_done = 0;
    double start = omp_get_wtime();
    classical_multiply();
    double time = omp_get_wtime() - start;
    
    printf("Classical method time: %.2f ms\n", time * 1000);
    printf("Result sample C[0][0] = %.2f\n\n", C_classical[0][0]);
}

void speculative_execution() {
    printf("=== Speculative Execution ===\n");
    
    classical_done = 0;
    strassen_done = 0;
    double total_start = omp_get_wtime();
    
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            classical_multiply();
        }
        
        #pragma omp section
        {
            strassen_multiply_simple();
        }
    }
    
    double total_time = omp_get_wtime() - total_start;
    
    printf("Classical method time: %.2f ms\n", classical_time * 1000);
    printf("Strassen method time: %.2f ms\n", strassen_time * 1000);
    printf("Total parallel time: %.2f ms\n", total_time * 1000);
    
    if (strassen_time < classical_time) {
        printf("\nSelected: Strassen's method (faster)\n");
        printf("Result sample C[0][0] = %.2f\n", C_strassen[0][0]);
        printf("Wasted computation: Classical method (%.2f ms, %.1f%%)\n", 
               classical_time * 1000, (classical_time / total_time) * 100);
    } else {
        printf("\nSelected: Classical method (faster/simpler)\n");
        printf("Result sample C[0][0] = %.2f\n", C_classical[0][0]);
        printf("Wasted computation: Strassen method (%.2f ms, %.1f%%)\n", 
               strassen_time * 1000, (strassen_time / total_time) * 100);
    }
}

int main() {
    printf("=== Matrix Multiplication: Speculative Decomposition ===\n");
    printf("Matrix size: %dx%d\n\n", N, N);
    
    matrix_init();
    
    sequential_execution();
    speculative_execution();
    
    printf("\n=== Performance Summary ===\n");
    printf("Sequential approach: Choose one method, run it\n");
    printf("Speculative approach: Run both, select faster result\n");
    printf("Trade-off: Lower latency vs wasted computation\n");
    
    return 0;
}