#include "kernel.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA) __global__
    void seluKernel(const int n, const float lambd, const float* input, float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        float scale = 1.05f;
        float alpha = 1.67326f;

        output[i] = scale * (max(0.0f, input[i]) + min(0.0f, alpha * (exp(input[i]) - 1)));
    }
}

pluginStatus_t seluGPU(cudaStream_t stream, const int n, const float lambd, const void* input, void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    seluKernel<BS><<<GS, BS, 0, stream>>>(n, lambd,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

pluginStatus_t seluInference(
    cudaStream_t stream, const int n, const float lambd, const void* input, void* output)
{
    return seluGPU(stream, n, lambd, (const float*) input, (float*) output);
}
