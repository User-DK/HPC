#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

__global__ void matrixMul(const float *A, const float *B, float *C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < P) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

inline void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    int N = 1000;
    if (argc >= 2) N = atoi(argv[1]);
    int M = N, P = N;

    // Host allocation
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(N * P * sizeof(float));
    float *h_C = (float*)malloc(M * P * sizeof(float));
    float *h_C_gpu = (float*)malloc(M * P * sizeof(float));
    if (!h_A || !h_B || !h_C || !h_C_gpu) { perror("malloc"); return 1; }

    // init
    srand(42);
    for (int i = 0; i < M * N; ++i) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < N * P; ++i) h_B[i] = rand() / (float)RAND_MAX;

    // Serial CPU execution
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * P + j];
            }
            h_C[i * P + j] = sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc((void**)&d_A, M * N * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, N * P * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, M * P * sizeof(float)), "cudaMalloc d_C");

    // copy host -> device
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");
    cudaEvent_t start_kernel, stop_kernel;
    checkCuda(cudaEventCreate(&start_kernel), "eventCreate start_kernel");
    checkCuda(cudaEventCreate(&stop_kernel), "eventCreate stop_kernel");
    checkCuda(cudaEventRecord(start), "eventRecord start");

    checkCuda(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, N * P * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    // launch kernel
    dim3 threads(16, 16);
    dim3 blocks((P + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    checkCuda(cudaEventRecord(start_kernel), "eventRecord start_kernel");
    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C, M, N, P);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop_kernel), "eventRecord stop_kernel");

    // copy device -> host
    checkCuda(cudaMemcpy(h_C_gpu, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSynchronize stop");
    float gpu_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "eventElapsedTime");
    float kernel_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel), "eventElapsedTime kernel");

    // verify results (small tolerance)
    double max_err = 0.0;
    for (int i = 0; i < M * P; ++i) {
        double err = fabs(h_C[i] - h_C_gpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("N = %d (M=N=P)\n", N);
    printf("CPU time  : %.3f ms\n", cpu_ms);
    printf("GPU time (H2D + kernel + D2H) : %.3f ms\n", gpu_ms);
    printf("GPU kernel time : %.3f ms\n", kernel_ms);
    printf("Speedup (CPU/GPU) : %.2f\n", cpu_ms / gpu_ms);
    printf("max error = %.6e\n", max_err);
    printf("sample result C[0][0] CPU=%.6f GPU=%.6f\n", h_C[0], h_C_gpu[0]);

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_gpu);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
    return 0;
}