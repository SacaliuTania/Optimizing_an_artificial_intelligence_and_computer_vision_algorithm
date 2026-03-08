// // yolo_cuda_kernels.cu
// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>
// #include <math.h>
// #include <algorithm>
//
// using namespace std;
//
//
// __global__ void ApplyMaskKernel(
//     const float* protos,      // [num_protos x H*W]
//     const float* coeffs,      // [num_protos] pentru obiectul curent
//     unsigned char* mask_out,  // [H*W] masca binară rezultată
//     int num_protos,
//     int H,
//     int W,
//     float threshold
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= H * W) return;
//
//     float val = 0.0f;
//     for (int p = 0; p < num_protos; ++p) {
//         val += coeffs[p] * protos[p * H * W + idx];
//     }
//
//     val = 1.0f / (1.0f + expf(-val)); // sigmoid
//     mask_out[idx] = (val > threshold) ? 255 : 0;
// }
//
// // Funție de lansare kernel segmentare
// extern "C" void launchMaskKernel(
//     const float* protos,
//     const float* coeffs,
//     unsigned char* mask_out,
//     int num_protos,
//     int H,
//     int W,
//     float threshold = 0.5f
// ) {
//     int threads = 256;
//     int blocks = (H * W + threads - 1) / threads;
//     ApplyMaskKernel<<<blocks, threads>>>(protos, coeffs, mask_out, num_protos, H, W, threshold);
//     cudaDeviceSynchronize();
// }
//
// __global__ void ApplyBBoxKernel(
//     const float* boxes_in,    // [num_boxes x 4] (x, y, w, h YOLO)
//     unsigned char* boxes_out, // [num_boxes x 4] (left, top, right, bottom)
//     int num_boxes,
//     int imgH,
//     int imgW,
//     float scale
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_boxes) return;
//
//     float x = boxes_in[idx * 4 + 0];
//     float y = boxes_in[idx * 4 + 1];
//     float w = boxes_in[idx * 4 + 2];
//     float h = boxes_in[idx * 4 + 3];
//
//     int left   = max(0, min(imgW - 1, int((x - 0.5f * w) * scale)));
//     int top    = max(0, min(imgH - 1, int((y - 0.5f * h) * scale)));
//     int right  = max(0, min(imgW - 1, int((x + 0.5f * w) * scale)));
//     int bottom = max(0, min(imgH - 1, int((y + 0.5f * h) * scale)));
//
//     boxes_out[idx * 4 + 0] = left;
//     boxes_out[idx * 4 + 1] = top;
//     boxes_out[idx * 4 + 2] = right;
//     boxes_out[idx * 4 + 3] = bottom;
// }
//
// extern "C" void launchBBoxKernel(
//     const float* boxes_in,
//     unsigned char* boxes_out,
//     int num_boxes,
//     int imgH,
//     int imgW,
//     float scale
// ) {
//     int threads = 256;
//     int blocks = (num_boxes + threads - 1) / threads;
//     ApplyBBoxKernel<<<blocks, threads>>>(boxes_in, boxes_out, num_boxes, imgH, imgW, scale);
//     cudaDeviceSynchronize();
// }

// kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <algorithm>

using namespace std;

__global__ void ApplyMaskKernel(
    const float* protos,      // [num_protos x H*W]
    const float* coeffs,      // [num_protos] pentru obiectul curent
    unsigned char* mask_out,  // [H*W] masca
    int num_protos,
    int H,
    int W,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * W) return;

    float val = 0.0f;
    for (int p = 0; p < num_protos; ++p) {
        val += coeffs[p] * protos[p * H * W + idx];
    }

    val = 1.0f / (1.0f + expf(-val)); // sigmoid
    mask_out[idx] = (val > threshold) ? 255 : 0;
}

extern "C" void launchMaskKernel(
    const float* protos,
    const float* coeffs,
    unsigned char* mask_out,
    int num_protos,
    int H,
    int W,
    float threshold = 0.5f
) {
    int threads = 256;
    int blocks = (H * W + threads - 1) / threads;
    ApplyMaskKernel<<<blocks, threads>>>(protos, coeffs, mask_out, num_protos, H, W, threshold);
    cudaDeviceSynchronize();
}

__global__ void ApplyBBoxKernel(
    const float* boxes_in,    // [num_boxes x 4] (x, y, w, h YOLO)
    unsigned char* boxes_out, // [num_boxes x 4] (left, top, right, bottom)
    int num_boxes,
    int imgH,
    int imgW,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    float x = boxes_in[idx * 4 + 0];
    float y = boxes_in[idx * 4 + 1];
    float w = boxes_in[idx * 4 + 2];
    float h = boxes_in[idx * 4 + 3];

    int left   = max(0, min(imgW - 1, int((x - 0.5f * w) * scale)));
    int top    = max(0, min(imgH - 1, int((y - 0.5f * h) * scale)));
    int right  = max(0, min(imgW - 1, int((x + 0.5f * w) * scale)));
    int bottom = max(0, min(imgH - 1, int((y + 0.5f * h) * scale)));

    boxes_out[idx * 4 + 0] = left;
    boxes_out[idx * 4 + 1] = top;
    boxes_out[idx * 4 + 2] = right;
    boxes_out[idx * 4 + 3] = bottom;
}

extern "C" void launchBBoxKernel(
    const float* boxes_in,
    unsigned char* boxes_out,
    int num_boxes,
    int imgH,
    int imgW,
    float scale
) {
    int threads = 256;
    int blocks = (num_boxes + threads - 1) / threads;
    ApplyBBoxKernel<<<blocks, threads>>>(boxes_in, boxes_out, num_boxes, imgH, imgW, scale);
    cudaDeviceSynchronize();
}
