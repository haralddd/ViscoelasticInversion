#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <curand.h>
#include <curand_kernel.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// cuRAND error checking macro
#define CURAND_CHECK(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Finite difference stencil kernel for x-direction
__global__ void ddx_kernel(float* du, const float* u, const int* grid, const float* coefs, 
                          int M, int N, int stencil_size, int pad) {
    int m = blockIdx.y * blockDim.y + threadIdx.y + pad;
    int n = blockIdx.x * blockDim.x + threadIdx.x + pad;
    
    if (m >= M - pad || n >= N - pad) return;
    
    float val = 0.0f;
    for (int i = 0; i < stencil_size; i++) {
        int g = grid[i];
        float c = coefs[i];
        val += c * u[(m + g) * N + n];
    }
    du[m * N + n] = val;
}

// Finite difference stencil kernel for z-direction
__global__ void ddz_kernel(float* du, const float* u, const int* grid, const float* coefs,
                          int M, int N, int stencil_size, int pad) {
    int m = blockIdx.y * blockDim.y + threadIdx.y + pad;
    int n = blockIdx.x * blockDim.x + threadIdx.x + pad;
    
    if (m >= M - pad || n >= N - pad) return;
    
    float val = 0.0f;
    for (int i = 0; i < stencil_size; i++) {
        int g = grid[i];
        float c = coefs[i];
        val += c * u[m * N + (n + g)];
    }
    du[m * N + n] = val;
}

// Generate central difference coefficients for 8th order first derivative
std::vector<float> generate_central_coeffs(int order, float dx) {
    // Hardcoded coefficients for 8th order central difference first derivative
    // These are the standard coefficients: [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]
    std::vector<float> coeffs = {
        1.0f/280.0f, -4.0f/105.0f, 1.0f/5.0f, -4.0f/5.0f, 
        0.0f, 4.0f/5.0f, -1.0f/5.0f, 4.0f/105.0f, -1.0f/280.0f
    };
    
    // Scale by 1/dx
    for (float& c : coeffs) {
        c /= dx;
    }
    
    return coeffs;
}

int main() {
    // Parameters matching Julia implementation
    const int M = 4096;
    const int N = 4096;
    const int diff_order = 8;
    const int iters = 1000;
    const int M_in = M - diff_order;
    const int N_in = N - diff_order;
    const int stencil_size = diff_order + 1;
    const int pad = diff_order / 2;
    
    // Generate stencil coefficients
    std::vector<float> coefs = generate_central_coeffs(diff_order, 1.0f);
    std::vector<int> grid(stencil_size);
    for (int i = 0; i < stencil_size; i++) {
        grid[i] = i - pad;
    }
    
    // Allocate device memory
    float *d_u, *d_du, *d_du2;
    int *d_grid, *d_grid2;
    float *d_coefs, *d_coefs2;
    
    CUDA_CHECK(cudaMalloc(&d_u, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_du, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_du2, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid, stencil_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid2, stencil_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coefs, stencil_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coefs2, stencil_size * sizeof(float)));
    
    // Copy stencil data to device
    CUDA_CHECK(cudaMemcpy(d_grid, grid.data(), stencil_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, grid.data(), stencil_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coefs, coefs.data(), stencil_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coefs2, coefs.data(), stencil_size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Setup kernel launch parameters
    dim3 blockDim(16, 16);
    dim3 gridDim((N_in + blockDim.x - 1) / blockDim.x, (M_in + blockDim.y - 1) / blockDim.y);
    
    // Create cuRAND generator
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, time(NULL)));
    
    // Timing vectors
    std::vector<double> gxts(iters), gzts(iters);
    
    // Benchmark ddx kernel
    for (int i = 0; i < iters; i++) {
        // Generate random input on device
        CURAND_CHECK(curandGenerateUniform(generator, d_u, M * N));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        ddx_kernel<<<gridDim, blockDim>>>(d_du, d_u, d_grid, d_coefs, M, N, stencil_size, pad);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        gxts[i] = elapsed.count();
    }
    
    // Benchmark ddz kernel
    for (int i = 0; i < iters; i++) {
        // Generate random input on device
        CURAND_CHECK(curandGenerateUniform(generator, d_u, M * N));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        ddz_kernel<<<gridDim, blockDim>>>(d_du2, d_u, d_grid2, d_coefs2, M, N, stencil_size, pad);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        gzts[i] = elapsed.count();
    }
    
    // Calculate statistics
    std::sort(gxts.begin(), gxts.end());
    std::sort(gzts.begin(), gzts.end());
    
    double gxt = gxts[iters / 2]; // median
    double gzt = gzts[iters / 2]; // median
    
    double gxt_var = 0.0, gzt_var = 0.0;
    for (int i = 0; i < iters; i++) {
        gxt_var += (gxts[i] - gxt) * (gxts[i] - gxt);
        gzt_var += (gzts[i] - gzt) * (gzts[i] - gzt);
    }
    gxt_var /= iters;
    gzt_var /= iters;
    
    // Print results
    std::cout << "GPU diff x: " << gxt * 1e3 << " ms (σ = " << std::sqrt(gxt_var) * 1e3 << ")" << std::endl;
    std::cout << "GPU diff z: " << gzt * 1e3 << " ms (σ = " << std::sqrt(gzt_var) * 1e3 << ")" << std::endl;
    
    // Cleanup
    CURAND_CHECK(curandDestroyGenerator(generator));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_du));
    CUDA_CHECK(cudaFree(d_du2));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaFree(d_coefs));
    CUDA_CHECK(cudaFree(d_coefs2));
    
    return 0;
}
