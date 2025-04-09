import torch
import triton
import triton.language as tl
from torch import Tensor

# --------------------
# Forward kernel
# --------------------
# The forward kernel computes for each element:
#   P = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5   (computed by Horner’s method)
#   Q = 1 + |b0|*|x| + |b1|*|x|^2 + |b2|*|x|^3 + |b3|*|x|^4
#   result = P / Q
#
# Each “group” uses 6 coefficients from a and 4 coefficients from b.
#
# We assume the following inputs:
#   x_ptr: pointer to input tensor (flattened, size = B*L*D)
#   a_ptr: pointer to numerator coefficients (per–group, groups = group count)
#   b_ptr: pointer to denominator coefficients (per–group)
#   result_ptr: pointer to output tensor (flattened)
#   x_size: total number of elements
#   D: size of the last dimension
#   D_per_group: D divided by the number of groups
#
# The grid is 1D.
@triton.jit
def rational_fwd_kernel(
    x_ptr, a_ptr, b_ptr, result_ptr,
    D, group, x_size, D_per_group,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Load input elements.
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Determine d index and group index.
    d_index = offs % D
    g_index = d_index // D_per_group

    # Compute coefficient offsets.
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load numerator coefficients.
    s_a0 = tl.load(a_ptr + a_offset + 0)
    s_a1 = tl.load(a_ptr + a_offset + 1)
    s_a2 = tl.load(a_ptr + a_offset + 2)
    s_a3 = tl.load(a_ptr + a_offset + 3)
    s_a4 = tl.load(a_ptr + a_offset + 4)
    s_a5 = tl.load(a_ptr + a_offset + 5)

    # Load denominator coefficients (using absolute value).
    s_b0 = tl.abs(tl.load(b_ptr + b_offset + 0))
    s_b1 = tl.abs(tl.load(b_ptr + b_offset + 1))
    s_b2 = tl.abs(tl.load(b_ptr + b_offset + 2))
    s_b3 = tl.abs(tl.load(b_ptr + b_offset + 3))

    abs_x = tl.abs(x_val)

    # Compute numerator polynomial P(x) via Horner's method.
    P = s_a5
    P = tl.fma(P, x_val, s_a4)
    P = tl.fma(P, x_val, s_a3)
    P = tl.fma(P, x_val, s_a2)
    P = tl.fma(P, x_val, s_a1)
    P = tl.fma(P, x_val, s_a0)

    # Compute denominator polynomial Q(x).
    Q = s_b3
    Q = tl.fma(Q, abs_x, s_b2)
    Q = tl.fma(Q, abs_x, s_b1)
    Q = tl.fma(Q, abs_x, s_b0)
    Q = tl.fma(Q, abs_x, 1.0)

    tl.store(result_ptr + offs, P / Q, mask=mask)

def rational_fwd_triton(x, n, d, group):
    D = x.shape[-1]
    x_size = x.numel()
    D_per_group = D // group

    result = torch.empty_like(x)
    BLOCK_SIZE = 256
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_fwd_kernel[(num_blocks,)](
        x, n, d, result,
        D, group, x_size, D_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return result

# --------------------
# Backward kernel
# --------------------
# The backward kernel computes gradients with respect to the input x and the coefficients.
# For each element it computes:
#
#   xp = x
#   axp = |x|
#   P = a0 + a1*x + a2*x^2 + ... + a5*x^5
#   Q = 1 + |b0|*axp + |b1|*axp^2 + |b2|*axp^3 + |b3|*axp^4
#   R = a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 + 5*a5*x^4
#   S = sign(x) * (|b0| + 2*|b1|*axp + 3*|b2|*axp^2 + 4*|b3|*axp^3)
#
# and then:
#   d_x = (R/Q + S * (-P/(Q^2))) * grad_o
#
# It also computes per–coefficient gradients:
#
#   d_a[0] = grad_o/Q,  d_a[i] = (x^i * grad_o)/Q, for i = 1,...,5
#   d_b[i] = (-P/(Q^2)) * sign(b[i]) * (axp^(i+1)) * grad_o, for i = 0,...,3
#
# The results for d_a and d_b are accumulated via atomic adds.
@triton.jit
def rational_bwd_kernel(
    grad_output_ptr, x_ptr, a_ptr, b_ptr,
    d_x_ptr, d_a_ptr, d_b_ptr,
    D, group, x_size, n_size, d_size, D_per_group,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < x_size

    # Load grad_output and x.
    grad_o = tl.load(grad_output_ptr + offs, mask=mask)
    x_val = tl.load(x_ptr + offs, mask=mask)

    # Determine group index
    d_index = offs % D
    g_index = d_index // D_per_group
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load coefficients for a.
    a0 = tl.load(a_ptr + a_offset + 0)
    a1 = tl.load(a_ptr + a_offset + 1)
    a2 = tl.load(a_ptr + a_offset + 2)
    a3 = tl.load(a_ptr + a_offset + 3)
    a4 = tl.load(a_ptr + a_offset + 4)
    a5 = tl.load(a_ptr + a_offset + 5)

    # Load coefficients for b (and compute their absolute values).
    b0 = tl.load(b_ptr + b_offset + 0)
    b1 = tl.load(b_ptr + b_offset + 1)
    b2 = tl.load(b_ptr + b_offset + 2)
    b3 = tl.load(b_ptr + b_offset + 3)
    b0_abs = tl.abs(b0)
    b1_abs = tl.abs(b1)
    b2_abs = tl.abs(b2)
    b3_abs = tl.abs(b3)

    # Compute powers of x.
    xp = x_val
    xp2 = xp * xp
    xp3 = xp2 * xp
    xp4 = xp3 * xp
    xp5 = xp4 * xp

    # Compute absolute value of x and its powers.
    axp = tl.abs(x_val)
    axp2 = axp * axp
    axp3 = axp2 * axp
    axp4 = axp3 * axp

    # Compute P, Q, R, S.
    P = a0 + a1 * xp + a2 * xp2 + a3 * xp3 + a4 * xp4 + a5 * xp5
    Q = 1.0 + b0_abs * axp + b1_abs * axp2 + b2_abs * axp3 + b3_abs * axp4
    R = a1 + 2.0*a2 * xp + 3.0*a3 * xp2 + 4.0*a4 * xp3 + 5.0*a5 * xp4
    # Compute sign(x): if x<0 then -1, else 1.
    sign_x = tl.where(x_val < 0, -1.0, 1.0)
    S = sign_x * (b0_abs + 2.0*b1_abs * axp + 3.0*b2_abs * axp2 + 4.0*b3_abs * axp3)

    mpq2 = -P / (Q * Q)

    # Compute gradient for x.
    dx = (R / Q + S * mpq2) * grad_o
    tl.store(d_x_ptr + offs, dx, mask=mask)

    # Compute gradients for a.
    da0 = grad_o / Q
    da1 = xp * grad_o / Q
    da2 = xp2 * grad_o / Q
    da3 = xp3 * grad_o / Q
    da4 = xp4 * grad_o / Q
    da5 = xp5 * grad_o / Q

    # Compute gradients for b.
    # Note: for each coefficient b_i, we use the original sign.
    sign_b0 = tl.where(b0 < 0, -1.0, 1.0)
    sign_b1 = tl.where(b1 < 0, -1.0, 1.0)
    sign_b2 = tl.where(b2 < 0, -1.0, 1.0)
    sign_b3 = tl.where(b3 < 0, -1.0, 1.0)
    db0 = mpq2 * sign_b0 * axp * grad_o
    db1 = mpq2 * sign_b1 * axp2 * grad_o
    db2 = mpq2 * sign_b2 * axp3 * grad_o
    db3 = mpq2 * sign_b3 * axp4 * grad_o

    # Accumulate contributions for coefficients for a.
    tl.atomic_add(d_a_ptr + (a_offset + 0), da0, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 1), da1, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 2), da2, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 3), da3, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 4), da4, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 5), da5, mask=mask)

    # Accumulate contributions for coefficients for b.
    tl.atomic_add(d_b_ptr + (b_offset + 0), db0, mask=mask)
    tl.atomic_add(d_b_ptr + (b_offset + 1), db1, mask=mask)
    tl.atomic_add(d_b_ptr + (b_offset + 2), db2, mask=mask)
    tl.atomic_add(d_b_ptr + (b_offset + 3), db3, mask=mask)
        
def rational_bwd_triton(grad_output, x, n, d, group):
    D = x.shape[-1]
    x_size = x.numel()
    n_size = n.numel()
    d_size = d.numel()
    D_per_group = D // group

    d_x = torch.empty_like(x)
    d_n = torch.zeros_like(n, dtype=torch.float32)
    d_d = torch.zeros_like(d, dtype=torch.float32)

    BLOCK_SIZE = 256
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_bwd_kernel[(num_blocks,)](
        grad_output, x, n, d,
        d_x, d_n, d_d,
        D, group, x_size, n_size, d_size, D_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return d_x, d_n, d_d


class RationalTriton1DGroup(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx: torch.autograd.Function, 
                input: Tensor, 
                weight_numerator: Tensor, 
                weight_denominator: Tensor, 
                group: int) -> Tensor:
        """
        Forward pass of the rational function computed with Triton kernels.
        
        Args:
            ctx: The context object for storing information for the backward pass.
            input (Tensor): Input tensor.
            weight_numerator (Tensor): Weights for the numerator polynomial.
            weight_denominator (Tensor): Weights for the denominator polynomial.
            group (int): The group number (non-differentiable).
        
        Returns:
            Tensor: Output tensor resulting from applying the rational function.
        """
        # Save tensors required for backward pass.
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group

        # Compute the forward pass using a Triton kernel.
        output = rational_fwd_triton(input, weight_numerator, weight_denominator, group)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx: torch.autograd.Function, grad_output: Tensor):
        """
        Backward pass of the rational function computed with Triton kernels.
        
        Args:
            ctx: The context object with saved tensors.
            grad_output (Tensor): Gradient of the loss with respect to the output.
        
        Returns:
            Tuple[Tensor, Tensor, Tensor, None]:
                - Gradient with respect to the input.
                - Gradient with respect to weight_numerator.
                - Gradient with respect to weight_denominator.
                - None for the non-differentiable 'group' parameter.
        """
        # Retrieve saved tensors and the group number.
        input, weight_numerator, weight_denominator = ctx.saved_tensors
        group = ctx.group

        # Compute gradients using a Triton backward kernel.
        d_input, d_weight_numerator, d_weight_denominator = rational_bwd_triton(
            grad_output, input, weight_numerator, weight_denominator, group
        )

        # Return gradients. None is returned for 'group' as it is non-differentiable.
        return d_input, d_weight_numerator, d_weight_denominator, None
