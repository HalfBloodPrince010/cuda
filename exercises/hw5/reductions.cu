
#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const size_t N = 2*8ULL*1024ULL*1024ULL;  // data size
//const size_t N = 256*640; // data size
const int BLOCK_SIZE = 256;  // CUDA maximum is 1024
// naive atomic reduction kernel
__global__ void atomic_red(const float *gdata, float *out){
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N) atomicAdd(out, gdata[idx]);
}

__global__ void reduce(float *gdata, float *out){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < N) {  // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += gridDim.x*blockDim.x;
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // parallel sweep reduction
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) out[blockIdx.x] = sdata[0];
  }

 __global__ void reduce_a(float *gdata, float *out){
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     size_t idx = threadIdx.x+blockDim.x*blockIdx.x;

     while (idx < N) {  // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += gridDim.x*blockDim.x;
        }

     for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        __syncthreads();
        if (tid < s)  // parallel sweep reduction
            sdata[tid] += sdata[tid + s];
        }
     if (tid == 0) atomicAdd(out, sdata[0]);
  }


__global__ void reduce_warp_shuffle(float *gdata, float *out) {
        // Each Warp Produces a value. Utmost 1024/32 warps = 32 warps
        __shared__ float sdata[32];
        int tid = threadIdx.x;
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        // each thread has a val
        float val = 0.0f;
        unsigned mask = 0xFFFFFFFFU;
        int lane = threadIdx.x % warpSize;
        int warpID = threadIdx.x / warpSize;

        //grid stride loop to handle variable length data
        while(idx < N) {
            val += gdata[idx];
            idx += gridDim.x * blockDim.x;  // Total threads in the grid
        }

        // 1st warp-shuffle operation, where each warp produces a value (tid/lane = 0)
        // holds that value
        for(int offset = warpSize/2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if(lane == 0) sdata[warpID] = val;
        __syncthreads();

        // Now we have to add 32 elements, hence we just need 1 warp
        if(warpID == 0) {
            // this condition ensures we only consider the warp that participated.
            // For instance if there were only 512 threads, then 512/32 = 16
            // Only 16 warps produced the value. Hence we don't consider other 16 warps for addition.
            val = (tid < blockDim.x/warpSize)?sdata[lane]:0;

            // Final Warp Shuffle
            for(int offset = warpSize/2; offset > 0; offset >>= 1){
                val += __shfl_down_sync(mask, val, offset);
            }

            // Produced 1 value for block
            // All block after final warp shuffle atomically adds to the out.
            // This prevents another reduction.
            if(tid == 0) atomicAdd(out, val);
        }
    }


int main(){

  float *h_A, *h_sum, *d_A, *d_sum;
  h_A = new float[N];  // allocate space for data in host memory
  h_sum = new float;
  for (int i = 0; i < N; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, N*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sum, sizeof(float));  // allocate device space for sum
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  atomic_red<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("atomic reduction kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("atomic sum reduction incorrect!\n"); return -1;}
  printf("atomic sum reduction correct!\n");
  const int blocks = 640;
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction w/atomic kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction w/atomic sum incorrect!\n"); return -1;}
  printf("reduction w/atomic sum correct!\n");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_warp_shuffle<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction warp shuffle kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction warp shuffle sum incorrect!\n"); return -1;}
  printf("reduction warp shuffle sum correct!\n");
  return 0;
}
