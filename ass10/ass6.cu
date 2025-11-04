#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

__global__ void matrixAdd(const float *A, const float *B, float *C, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

inline void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    int M = 1000, N = 1000;
    if (argc >= 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }

    // Host allocation
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(M * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *h_C_gpu = (float*)malloc(M * N * sizeof(float));
    if (!h_A || !h_B || !h_C || !h_C_gpu) { perror("malloc"); return 1; }

    // init
    srand(42);
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Serial CPU execution
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            h_C[idx] = h_A[idx] + h_B[idx];
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU allocation
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc((void**)&d_A, M * N * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, M * N * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, M * N * sizeof(float)), "cudaMalloc d_C");

    // copy host -> device
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");
    cudaEvent_t start_kernel, stop_kernel;
    checkCuda(cudaEventCreate(&start_kernel), "eventCreate start_kernel");
    checkCuda(cudaEventCreate(&stop_kernel), "eventCreate stop_kernel");
    checkCuda(cudaEventRecord(start), "eventRecord start");

    checkCuda(cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, M * N * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    // launch kernel
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    checkCuda(cudaEventRecord(start_kernel), "eventRecord start_kernel");
    matrixAdd<<<blocks, threads>>>(d_A, d_B, d_C, M, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop_kernel), "eventRecord stop_kernel");

    // copy device -> host
    checkCuda(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSynchronize stop");
    float gpu_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "eventElapsedTime");
    float kernel_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel), "eventElapsedTime kernel");

    // verify results (small tolerance)
    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double err = fabs(h_C[i] - h_C_gpu[i]);
        if (err > max_err) max_err = err;
    }

    printf("M = %d, N = %d\n", M, N);
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