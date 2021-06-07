//
// Created by kouushou on 2021/6/6.
//
#include "Mytime.h"
#include <cstdio>

const int blockSIZE = 512;

__global__ void reductionNeighbored(double *idata, double *outdata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double *curData = idata + blockIdx.x * blockDim.x;
    if (idx >= n)return;

    for (unsigned stride = 1; stride < blockDim.x; stride <<= 1u) {

        if (!(tid % (stride << 1u))) {
            curData[tid] += curData[stride + tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        outdata[blockIdx.x] = curData[0];
    }
}

double calReductionGPU_Neighbored(const double *a, unsigned siz, double *time) {
    double *d_val;
    double *d_val_OUT;
    double *h_val_OUT;
    unsigned int nnz = 1u << siz;
    dim3 block(blockSIZE, 1);
    dim3 grid((nnz + blockSIZE - 1) / blockSIZE, 1);

    h_val_OUT = (double *) malloc(sizeof(double) * grid.x);

    cudaMalloc(&d_val_OUT, sizeof(double) * grid.x);
    cudaMalloc(&d_val, sizeof(double) * nnz);

    cudaMemcpy(d_val, a, sizeof(double) * nnz, cudaMemcpyHostToDevice);


    MyTimeStart();
    reductionNeighbored<<<grid, block>>>(d_val, d_val_OUT, nnz);

    cudaDeviceSynchronize();
    *time = MyTimePassed();

    cudaMemcpy(h_val_OUT, d_val_OUT, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double res = 0;
    for (int i = 0; i < grid.x; ++i) {
        res += h_val_OUT[i];
    }
    free(h_val_OUT);
    cudaFree(d_val_OUT);
    cudaFree(d_val);
    return res;
}

__global__ void reductionNeighboredNoDivided(double *idata, double *outdata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double *curData = idata + blockIdx.x * blockDim.x;
    if (idx >= n)return;

    for (unsigned stride = 1; stride < blockDim.x; stride <<= 1u) {

        if (!(tid & ((stride << 1u) - 1u))) {
            curData[tid] += curData[stride + tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        outdata[blockIdx.x] = curData[0];
    }
}

double calReductionGPU_NeighboredNoDivided(const double *a, unsigned siz, double *time) {
    double *d_val;
    double *d_val_OUT;
    double *h_val_OUT;
    unsigned int nnz = 1u << siz;
    dim3 block(blockSIZE, 1);
    dim3 grid((nnz + blockSIZE - 1) / blockSIZE, 1);

    h_val_OUT = (double *) malloc(sizeof(double) * grid.x);

    cudaMalloc(&d_val_OUT, sizeof(double) * grid.x);
    cudaMalloc(&d_val, sizeof(double) * nnz);

    cudaMemcpy(d_val, a, sizeof(double) * nnz, cudaMemcpyHostToDevice);


    MyTimeStart();
    reductionNeighboredNoDivided<<<grid, block>>>(d_val, d_val_OUT, nnz);

    cudaDeviceSynchronize();
    *time = MyTimePassed();

    cudaMemcpy(h_val_OUT, d_val_OUT, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double res = 0;
    for (int i = 0; i < grid.x; ++i) {
        res += h_val_OUT[i];
    }
    free(h_val_OUT);
    cudaFree(d_val_OUT);
    cudaFree(d_val);
    return res;
}


__global__ void reductionNeighboredLessToRight(double *idata, double *outdata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double *curData = idata + blockIdx.x * blockDim.x;
    if (idx >= n)return;

    for (unsigned stride = 1; stride < blockDim.x; stride <<= 1u) {


        unsigned index = 2u * stride * tid;
        if (index < blockDim.x) {
            curData[index] += curData[stride + index];
        }

        __syncthreads();
    }
    if (tid == 0) {
        outdata[blockIdx.x] = curData[0];
    }
}

double calReductionGPU_NeighboredLessToRight(const double *a, unsigned siz, double *time) {
    double *d_val;
    double *d_val_OUT;
    double *h_val_OUT;
    unsigned int nnz = 1u << siz;
    dim3 block(blockSIZE, 1);
    dim3 grid((nnz + blockSIZE - 1) / blockSIZE, 1);

    h_val_OUT = (double *) malloc(sizeof(double) * grid.x);

    cudaMalloc(&d_val_OUT, sizeof(double) * grid.x);
    cudaMalloc(&d_val, sizeof(double) * nnz);

    cudaMemcpy(d_val, a, sizeof(double) * nnz, cudaMemcpyHostToDevice);


    MyTimeStart();
    reductionNeighboredLessToRight<<<grid, block>>>(d_val, d_val_OUT, nnz);

    cudaDeviceSynchronize();
    *time = MyTimePassed();

    cudaMemcpy(h_val_OUT, d_val_OUT, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double res = 0;
    for (int i = 0; i < grid.x; ++i) {
        res += h_val_OUT[i];
    }
    free(h_val_OUT);
    cudaFree(d_val_OUT);
    cudaFree(d_val);
    return res;
}

__global__ void reductionNeighboredReverse(double *idata, double *outdata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double *curData = idata + blockIdx.x * blockDim.x;
    if (idx >= n)return;

    for (unsigned stride = blockDim.x>>1u; stride; stride >>= 1u) {

        if(tid < stride){
            curData[tid]+=curData[tid+stride];
        }

        __syncthreads();
    }
    if (tid == 0) {
        outdata[blockIdx.x] = curData[0];
    }
}

double calReductionGPU_NeighboredReverse(const double *a, unsigned siz, double *time) {
    double *d_val;
    double *d_val_OUT;
    double *h_val_OUT;
    unsigned int nnz = 1u << siz;
    dim3 block(blockSIZE, 1);
    dim3 grid((nnz + blockSIZE - 1) / blockSIZE, 1);

    h_val_OUT = (double *) malloc(sizeof(double) * grid.x);

    cudaMalloc(&d_val_OUT, sizeof(double) * grid.x);
    cudaMalloc(&d_val, sizeof(double) * nnz);

    cudaMemcpy(d_val, a, sizeof(double) * nnz, cudaMemcpyHostToDevice);


    MyTimeStart();
    reductionNeighboredReverse<<<grid, block>>>(d_val, d_val_OUT, nnz);

    cudaDeviceSynchronize();
    *time = MyTimePassed();

    cudaMemcpy(h_val_OUT, d_val_OUT, sizeof(double) * grid.x, cudaMemcpyDeviceToHost);
    double res = 0;
    for (int i = 0; i < grid.x; ++i) {
        res += h_val_OUT[i];
    }
    free(h_val_OUT);
    cudaFree(d_val_OUT);
    cudaFree(d_val);
    return res;
}

