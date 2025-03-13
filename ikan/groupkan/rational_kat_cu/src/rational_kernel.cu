// #include <torch/extension.h>

// template <typename scalar_t>
// __global__ void rational_fwd_cuda_kernel(
//     const scalar_t* __restrict__ x, 
//     const scalar_t* __restrict__ a,
//     const scalar_t* __restrict__ b, 
//     scalar_t* __restrict__ result, 
//     size_t x_size) {

    
//     scalar_t a_0 = a[0];
    
//     scalar_t a_1 = a[1];
    
//     scalar_t a_2 = a[2];
    
//     scalar_t a_3 = a[3];
    
//     scalar_t a_4 = a[4];
    
//     scalar_t a_5 = a[5];
    
    
//     scalar_t ab_0 = abs(b[0]);
    
//     scalar_t ab_1 = abs(b[1]);
    
//     scalar_t ab_2 = abs(b[2]);
    
//     scalar_t ab_3 = abs(b[3]);
    
//     for (int index = blockIdx.x * blockDim.x + threadIdx.x;
//         index < x_size;
//         index += blockDim.x * gridDim.x){

//         scalar_t xp1 = x[index];
//         scalar_t abs_xp1 = abs(xp1);

//         // Horner's method for polynomial computation for P
//         scalar_t P = a_5;
//         P = P * xp1 + a_4;
//         P = P * xp1 + a_3;
//         P = P * xp1 + a_2;
//         P = P * xp1 + a_1;
//         P = P * xp1 + a_0;
        
//         // Horner's method for polynomial computation for Q
//         scalar_t Q = ab_3;
//         Q = Q * abs_xp1 + ab_2;
//         Q = Q * abs_xp1 + ab_1;
//         Q = Q * abs_xp1 + ab_0;
//         Q = Q * abs_xp1 + 1.0;

//         result[index] = P / Q;
//     }
// }

// template <typename scalar_t>
// __global__ void rational_fwd_cuda_kernel_optimized(
//     const scalar_t* __restrict__ x, 
//     const scalar_t* __restrict__ a,
//     const scalar_t* __restrict__ b, 
//     scalar_t* __restrict__ result, 
//     size_t x_size) {

//     // Use shared memory for coefficients
//     __shared__ scalar_t s_a[6];
//     __shared__ scalar_t s_ab[4];

//     if (threadIdx.x < 6) {
//         s_a[threadIdx.x] = a[threadIdx.x];
//     }
//     if (threadIdx.x < 4) {
//         s_ab[threadIdx.x] = abs(b[threadIdx.x]);
//     }
//     __syncthreads();

//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < x_size) {
//         scalar_t xp1 = x[index];
//         scalar_t abs_xp1 = abs(xp1);

//         // Horner's method for polynomial computation for P using shared memory
//         scalar_t P = s_a[5];
//         for (int i = 4; i >= 0; --i) {
//             P = fmaf(P, xp1, s_a[i]);  // using FMA
//         }
        
//         // Horner's method for polynomial computation for Q using shared memory
//         scalar_t Q = s_ab[3];
//         for (int i = 2; i >= 0; --i) {
//             Q = fmaf(Q, abs_xp1, s_ab[i]);  // using FMA
//         }
//         Q = fmaf(Q, abs_xp1, 1.0);

//         result[index] = P / Q;
//     }
// }

// torch::Tensor rational_fwd_cuda(
//     torch::Tensor x, 
//     torch::Tensor n, 
//     torch::Tensor d
//     ){
//     auto result = at::empty_like(x);
//     const auto x_size = x.numel();
//     int blockSize = 512;  // You might want to experiment with this value
//     int numBlocks = (x_size + blockSize - 1) / blockSize;

//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_fwd_cuda", ([&] {
//     rational_fwd_cuda_kernel<scalar_t>
//         <<<numBlocks, blockSize>>>(
//             x.data_ptr<scalar_t>(),
//             n.data_ptr<scalar_t>(),
//             d.data_ptr<scalar_t>(),
//             result.data_ptr<scalar_t>(),
//             x_size);
//         }));

//     return result;
// }

// torch::Tensor rational_fwd_cuda_optimized(
//     torch::Tensor x, 
//     torch::Tensor n, 
//     torch::Tensor d
//     ){
//     auto result = at::empty_like(x);
//     const auto x_size = x.numel();
//     int blockSize = 512;  // You might want to experiment with this value
//     int numBlocks = (x_size + blockSize - 1) / blockSize;

//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_fwd_cuda_optimized", ([&] {
//     rational_fwd_cuda_kernel_optimized<scalar_t>
//         <<<numBlocks, blockSize>>>(
//             x.data_ptr<scalar_t>(),
//             n.data_ptr<scalar_t>(),
//             d.data_ptr<scalar_t>(),
//             result.data_ptr<scalar_t>(),
//             x_size);
//         }));

//     return result;
// }

// //P(X) = a_0 + a_1*X + a_2*X^2 ...
// //Q(X) = 1 + |b_0||X| + |b_1||X|^2 + |b_2||X|^3
// //R(X) = a_1 + 2*a_2*X + 3*a_3*X ...
// //S(X) = sign(X) * ( |b_0| + 2|b_1||X| + 3|b_2||X|^2 ...)
// //dF/dx = (-P(X)/Q(X)^2)*S(X) + R(X)/Q(X)
// //dF/da_i = x^i/Q(X), i \in {0,5}
// //dF/db_i = (-P(X)/Q(X)^2) * sign(b_i) * |X^{i+1}| , i \in {0,4}

// template <typename scalar_t>
// __global__ void rational_bwd_cuda_kernel(
//     const scalar_t* __restrict__ grad_output,
//     const scalar_t* __restrict__ x,
//     const scalar_t* __restrict__ a,
//     const scalar_t* __restrict__ b,
//     scalar_t* __restrict__ d_x,
//     double* __restrict__ d_a,
//     double* __restrict__ d_b,
//     size_t x_size) {

//     __shared__ double sda[6];
//     __shared__ double sdb[4];

//     if( threadIdx.x == 0){
        
//         sda[0] = 0;
        
//         sda[1] = 0;
        
//         sda[2] = 0;
        
//         sda[3] = 0;
        
//         sda[4] = 0;
        
//         sda[5] = 0;
                
//         sdb[0] = 0;
        
//         sdb[1] = 0;
        
//         sdb[2] = 0;
        
//         sdb[3] = 0;
//     }

//     __syncthreads();
    
//     scalar_t d_a0 = 0;
//     scalar_t a_0 = a[0];
    
//     scalar_t d_a1 = 0;
//     scalar_t a_1 = a[1];
    
//     scalar_t d_a2 = 0;
//     scalar_t a_2 = a[2];
    
//     scalar_t d_a3 = 0;
//     scalar_t a_3 = a[3];
    
//     scalar_t d_a4 = 0;
//     scalar_t a_4 = a[4];
    
//     scalar_t d_a5 = 0;
//     scalar_t a_5 = a[5];
    
//     scalar_t d_b0 = 0;
//     scalar_t b_0 = b[0];
//     scalar_t ab_0 = abs(b_0);
    
//     scalar_t d_b1 = 0;
//     scalar_t b_1 = b[1];
//     scalar_t ab_1 = abs(b_1);
    
//     scalar_t d_b2 = 0;
//     scalar_t b_2 = b[2];
//     scalar_t ab_2 = abs(b_2);
    
//     scalar_t d_b3 = 0;
//     scalar_t b_3 = b[3];
//     scalar_t ab_3 = abs(b_3);
    
//     for (int index = blockIdx.x * blockDim.x + threadIdx.x;
//          index < x_size;
//          index += blockDim.x * gridDim.x)
//       {
//         scalar_t xp1 = x[index];
//         scalar_t axp1 = abs(xp1);

//         scalar_t xp2 = xp1 * xp1;
//         scalar_t axp2 = abs(xp2);
//         scalar_t xp3 = xp2 * xp1;
//         scalar_t axp3 = abs(xp3);
//         scalar_t xp4 = xp3 * xp1;
//         scalar_t axp4 = abs(xp4);
//         scalar_t xp5 = xp4 * xp1;
//         scalar_t axp5 = abs(xp5);
        
//         scalar_t P = a_0
//                 + a_1*xp1
//                 + a_2*xp2
//                 + a_3*xp3
//                 + a_4*xp4
//                 + a_5*xp5;

//         scalar_t Q = scalar_t(1.0)
//                 + ab_0 * axp1
//                 + ab_1 * axp2
//                 + ab_2 * axp3
//                 + ab_3 * axp4;

//         scalar_t R = a_1
//                 + scalar_t(2.0) * a_2 * xp1
//                 + scalar_t(3.0) * a_3 * xp2
//                 + scalar_t(4.0) * a_4 * xp3
//                 + scalar_t(5.0) * a_5 * xp4;

//         scalar_t S = copysign( scalar_t(1.0), xp1 ) * (ab_0
//                 + scalar_t(2.0) * ab_1 * axp1
//                 + scalar_t(3.0) * ab_2 * axp2
//                 + scalar_t(4.0) * ab_3 * axp3
//                 );

//         scalar_t mpq2 = -P/(Q*Q);

//         scalar_t grad_o = grad_output[index];

//         scalar_t d_i_x = (R/Q + S*mpq2);
//         d_x[index] = d_i_x * grad_o;

//         scalar_t d_i_b0 = mpq2 * copysign( scalar_t(1.0), b_0 ) * axp1;
//         d_b0 += d_i_b0 * grad_o;
//         scalar_t d_i_b1 = mpq2 * copysign( scalar_t(1.0), b_1 ) * axp2;
//         d_b1 += d_i_b1 * grad_o;
//         scalar_t d_i_b2 = mpq2 * copysign( scalar_t(1.0), b_2 ) * axp3;
//         d_b2 += d_i_b2 * grad_o;
//         scalar_t d_i_b3 = mpq2 * copysign( scalar_t(1.0), b_3 ) * axp4;
//         d_b3 += d_i_b3 * grad_o;
        
//         scalar_t d_i_a0 = scalar_t(1.0)/Q;
//         d_a0 += d_i_a0 * grad_o;

        
//         scalar_t d_i_a1  = xp1/Q;
//         d_a1 += d_i_a1 * grad_o;
        
//         scalar_t d_i_a2  = xp2/Q;
//         d_a2 += d_i_a2 * grad_o;
        
//         scalar_t d_i_a3  = xp3/Q;
//         d_a3 += d_i_a3 * grad_o;
        
//         scalar_t d_i_a4  = xp4/Q;
//         d_a4 += d_i_a4 * grad_o;
        
//         scalar_t d_i_a5  = xp5/Q;
//         d_a5 += d_i_a5 * grad_o;
//     }

    
//     atomicAdd(&sda[0], d_a0);
    
//     atomicAdd(&sda[1], d_a1);
    
//     atomicAdd(&sda[2], d_a2);
    
//     atomicAdd(&sda[3], d_a3);
    
//     atomicAdd(&sda[4], d_a4);
    
//     atomicAdd(&sda[5], d_a5);
        
//     atomicAdd(&sdb[0], d_b0);
    
//     atomicAdd(&sdb[1], d_b1);
    
//     atomicAdd(&sdb[2], d_b2);
    
//     atomicAdd(&sdb[3], d_b3);
    
//     __syncthreads();

//     if( threadIdx.x == 0){
        
//         atomicAdd(&d_a[0], sda[0]);
        
//         atomicAdd(&d_a[1], sda[1]);
        
//         atomicAdd(&d_a[2], sda[2]);
        
//         atomicAdd(&d_a[3], sda[3]);
        
//         atomicAdd(&d_a[4], sda[4]);
        
//         atomicAdd(&d_a[5], sda[5]);
                
//         atomicAdd(&d_b[0], sdb[0]);
        
//         atomicAdd(&d_b[1], sdb[1]);
        
//         atomicAdd(&d_b[2], sdb[2]);
        
//         atomicAdd(&d_b[3], sdb[3]);
//             }
// }

// template <typename scalar_t>
// __global__ void rational_bwd_cuda_kernel_optimized(
//     const scalar_t* __restrict__ grad_output,
//     const scalar_t* __restrict__ x,
//     const scalar_t* __restrict__ a,
//     const scalar_t* __restrict__ b,
//     scalar_t* __restrict__ d_x,
//     double* __restrict__ d_a,
//     double* __restrict__ d_b,
//     size_t x_size) {
    
//     __shared__ double sda[6];
//     __shared__ double sdb[4];
//     if (threadIdx.x == 0) {
//         sda[0] = 0;
//         sda[1] = 0;
//         sda[2] = 0;
//         sda[3] = 0;
//         sda[4] = 0;
//         sda[5] = 0;
//         sdb[0] = 0;
//         sdb[1] = 0;
//         sdb[2] = 0;
//         sdb[3] = 0;
//     }
    

//     __shared__ scalar_t shared_a[6];
//     __shared__ scalar_t shared_b_abs[4];
//     __shared__ scalar_t shared_b[4];

//     // Preloading 'a' coefficients into shared memory
//     if (threadIdx.x < 6) {
//         shared_a[threadIdx.x] = a[threadIdx.x];
//     }

//     // Preloading absolute values of 'b' coefficients into shared memory
//     if (threadIdx.x < 4) {
//         shared_b[threadIdx.x] = b[threadIdx.x];
//         shared_b_abs[threadIdx.x] = abs(shared_b[threadIdx.x]);
//     }

//     __syncthreads();

//     scalar_t local_da[6] = {0}; // Local accumulation arrays
//     scalar_t local_db[4] = {0};

//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < x_size) {
//         scalar_t xp = x[index];
//         scalar_t axp = abs(xp);
//         // Compute powers of xp
//         scalar_t xp_powers[5];
//         xp_powers[0] = xp;
//         xp_powers[1] = xp * xp_powers[0]; // xp^2
//         xp_powers[2] = xp * xp_powers[1]; // xp^3
//         xp_powers[3] = xp * xp_powers[2]; // xp^4
//         xp_powers[4] = xp * xp_powers[3]; // xp^5

//         // Compute powers of axp
//         scalar_t axp_powers[4];
//         axp_powers[0] = axp;
//         axp_powers[1] = axp * axp_powers[0]; // axp^2
//         axp_powers[2] = axp * axp_powers[1]; // axp^3
//         axp_powers[3] = axp * axp_powers[2]; // axp^4

//         // Compute absolute values once

//         scalar_t P = shared_a[0] 
//         + shared_a[1] * xp_powers[0] 
//         + shared_a[2] * xp_powers[1] 
//         + shared_a[3] * xp_powers[2] 
//         + shared_a[4] * xp_powers[3] 
//         + shared_a[5] * xp_powers[4];

//         scalar_t Q = scalar_t(1.0)
//         + shared_b_abs[0] * axp_powers[0] 
//         + shared_b_abs[1] * axp_powers[1] 
//         + shared_b_abs[2] * axp_powers[2] 
//         + shared_b_abs[3] * axp_powers[3];


//         scalar_t R = shared_a[1] 
//         + scalar_t(2.0) * shared_a[2] * xp_powers[0] 
//         + scalar_t(3.0) * shared_a[3] * xp_powers[1] 
//         + scalar_t(4.0) * shared_a[4] * xp_powers[2] 
//         + scalar_t(5.0) * shared_a[5] * xp_powers[3];

//         scalar_t S = copysign(scalar_t(1.0), xp) * (shared_b_abs[0] 
//         + scalar_t(2.0) * shared_b_abs[1] * axp_powers[0] 
//         + scalar_t(3.0) * shared_b_abs[2] * axp_powers[1] 
//         + scalar_t(4.0) * shared_b_abs[3] * axp_powers[2]);
        
//         // scalar_t Q_inv = 1.0 / Q;

//         scalar_t grad_o = grad_output[index];
        
//         scalar_t mpq2 = -P/(Q*Q);

//         scalar_t d_i_x = (R / Q + S * mpq2) * grad_o;
//         d_x[index] = d_i_x;

//         // Loop for computing d_a contributions
//         local_da[0] = scalar_t(1.0) / Q * grad_o;
//         for (int i = 1; i < 6; ++i) {
//             local_da[i] = (xp_powers[i-1] / Q) * grad_o;
//         }

//         // Loop for computing d_b contributions
//         for (int i = 0; i < 4; ++i) {
//             local_db[i] = mpq2 * copysign(scalar_t(1.0), shared_b[i]) * axp_powers[i] * grad_o;
//         }
//     }

//     // Reduce local arrays to shared memory
//     for (int i = 0; i < 6; ++i) {
//         atomicAdd(&sda[i], local_da[i]);
//     }
//     for (int i = 0; i < 4; ++i) {
//         atomicAdd(&sdb[i], local_db[i]);
//     }

//     __syncthreads();

//     // Only one thread writes back to global memory
//     if (threadIdx.x == 0) {
//         for (int i = 0; i < 6; ++i) {
//             atomicAdd(&d_a[i], sda[i]);
//         }
//         for (int i = 0; i < 4; ++i) {
//             atomicAdd(&d_b[i], sdb[i]);
//         }
//     }
// }

// std::vector<torch::Tensor> rational_bwd_cuda(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
//     const auto x_size = x.numel();
//     auto d_x = at::empty_like(x);
//     auto d_n = at::zeros_like(n).toType(at::kDouble);
//     auto d_d = at::zeros_like(d).toType(at::kDouble);

//     int blockSize = 512;  // You might want to experiment with this value
//     int numBlocks = (x_size + blockSize - 1) / blockSize;

//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_bwd_cuda", ([&] {
//     rational_bwd_cuda_kernel<scalar_t>
//         <<<numBlocks, blockSize>>>(
//             grad_output.data_ptr<scalar_t>(),
//             x.data_ptr<scalar_t>(),
//             n.data_ptr<scalar_t>(),
//             d.data_ptr<scalar_t>(),
//             d_x.data_ptr<scalar_t>(),
//             d_n.data_ptr<double>(),
//             d_d.data_ptr<double>(),
//             x_size);
//     }));

//     return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
// }

// std::vector<torch::Tensor> rational_bwd_cuda_optimized(torch::Tensor grad_output, torch::Tensor x, torch::Tensor n, torch::Tensor d){
//     const auto x_size = x.numel();
//     auto d_x = at::empty_like(x);
//     auto d_n = at::zeros_like(n).toType(at::kDouble);
//     auto d_d = at::zeros_like(d).toType(at::kDouble);

//     int blockSize = 128;  // You might want to experiment with this value
//     int numBlocks = (x_size + blockSize - 1) / blockSize;

//     AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rational_bwd_cuda_optimized", ([&] {
//     rational_bwd_cuda_kernel_optimized<scalar_t>
//         <<<numBlocks, blockSize>>>(
//             grad_output.data_ptr<scalar_t>(),
//             x.data_ptr<scalar_t>(),
//             n.data_ptr<scalar_t>(),
//             d.data_ptr<scalar_t>(),
//             d_x.data_ptr<scalar_t>(),
//             d_n.data_ptr<double>(),
//             d_d.data_ptr<double>(),
//             x_size);
//     }));

//     return {d_x, d_n.toType(at::kFloat), d_d.toType(at::kFloat)};
// }