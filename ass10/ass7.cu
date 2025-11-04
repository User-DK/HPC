#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

__global__ void dotProduct(const float *A, const float *B, float *sum, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    while (i < N) {
        local_sum += A[i] * B[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(sum, sdata[0]);
}

inline void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    int N = 10000000;
    if (argc >= 2) N = atoi(argv[1]);

    // Host allocation
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    if (!h_A || !h_B) { perror("malloc"); return 1; }

    // init
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Serial CPU execution
    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) cpu_sum += h_A[i] * h_B[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU allocation
    float *d_A = nullptr, *d_B = nullptr, *d_sum = nullptr;
    checkCuda(cudaMalloc((void**)&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, N * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void**)&d_sum, sizeof(float)), "cudaMalloc d_sum");

    // Initialize sum to 0
    checkCuda(cudaMemset(d_sum, 0, sizeof(float)), "cudaMemset d_sum");

    // copy host -> device
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "eventCreate start");
    checkCuda(cudaEventCreate(&stop), "eventCreate stop");
    cudaEvent_t start_kernel, stop_kernel;
    checkCuda(cudaEventCreate(&start_kernel), "eventCreate start_kernel");
    checkCuda(cudaEventCreate(&stop_kernel), "eventCreate stop_kernel");
    checkCuda(cudaEventRecord(start), "eventRecord start");

    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    // launch kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    int shared_mem_size = threads * sizeof(float);
    checkCuda(cudaEventRecord(start_kernel), "eventRecord start_kernel");
    dotProduct<<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_sum, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop_kernel), "eventRecord stop_kernel");

    // copy device -> host
    float gpu_sum = 0.0f;
    checkCuda(cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost), "D2H sum");

    checkCuda(cudaEventRecord(stop), "eventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "eventSynchronize stop");
    float gpu_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "eventElapsedTime");
    float kernel_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel), "eventElapsedTime kernel");

    // verify results (small tolerance)
    double err = fabs(cpu_sum - gpu_sum);

    printf("N = %d\n", N);
    printf("CPU time  : %.3f ms\n", cpu_ms);
    printf("GPU time (H2D + kernel + D2H) : %.3f ms\n", gpu_ms);
    printf("GPU kernel time : %.3f ms\n", kernel_ms);
    printf("Speedup (CPU/GPU) : %.2f\n", cpu_ms / gpu_ms);
    printf("CPU sum = %.6f\n", cpu_sum);
    printf("GPU sum = %.6f\n", gpu_sum);
    printf("error = %.6e\n", err);

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_sum);
    free(h_A); free(h_B);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
    return 0;
}