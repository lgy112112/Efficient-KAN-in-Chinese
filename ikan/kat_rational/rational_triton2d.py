import torch
import triton
import triton.language as tl
from torch import Tensor
# ==============================================================================
# 2D Forward Kernel: operates on a tensor of shape [B, D, H, W] in-place.
# The rational function is applied along the D dimension.
# ==============================================================================

@triton.jit
def rational_fwd_kernel_2d(
    x_ptr, a_ptr, b_ptr, result_ptr,
    B: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    group: tl.constexpr, x_size: tl.constexpr, D_per_group: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the global index.
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < x_size

    # Load the input value.
    x_val = tl.load(x_ptr + idx, mask=mask)

    # Compute the D coordinate for this element.
    # Total elements in one image: D*H*W.
    # Within each image, the flat index is idx % (D*H*W).
    d_index = (idx % (D * H * W)) // (H * W)
    # Compute the group index.
    g_index = d_index // D_per_group

    # Each group uses 6 numerator and 4 denominator coefficients.
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load numerator coefficients.
    s_a0 = tl.load(a_ptr + a_offset + 0)
    s_a1 = tl.load(a_ptr + a_offset + 1)
    s_a2 = tl.load(a_ptr + a_offset + 2)
    s_a3 = tl.load(a_ptr + a_offset + 3)
    s_a4 = tl.load(a_ptr + a_offset + 4)
    s_a5 = tl.load(a_ptr + a_offset + 5)

    # Load denominator coefficients (taking absolute values).
    s_b0 = tl.abs(tl.load(b_ptr + b_offset + 0))
    s_b1 = tl.abs(tl.load(b_ptr + b_offset + 1))
    s_b2 = tl.abs(tl.load(b_ptr + b_offset + 2))
    s_b3 = tl.abs(tl.load(b_ptr + b_offset + 3))

    # Compute absolute value of x.
    abs_x = tl.abs(x_val)

    # Evaluate numerator polynomial P(x) using Horner’s method.
    P = s_a5
    P = tl.fma(P, x_val, s_a4)
    P = tl.fma(P, x_val, s_a3)
    P = tl.fma(P, x_val, s_a2)
    P = tl.fma(P, x_val, s_a1)
    P = tl.fma(P, x_val, s_a0)

    # Evaluate denominator polynomial Q(x).
    Q = s_b3
    Q = tl.fma(Q, abs_x, s_b2)
    Q = tl.fma(Q, abs_x, s_b1)
    Q = tl.fma(Q, abs_x, s_b0)
    Q = tl.fma(Q, abs_x, 1.0)

    # Write the output.
    tl.store(result_ptr + idx, P / Q, mask=mask)


def rational_fwd_triton_2d(x: Tensor, n: Tensor, d: Tensor, group: int) -> Tensor:
    """
    2D forward helper.
    Expects x of shape [B, D, H, W] and applies the rational function along the D dimension.
    """
    B, D, H, W = x.shape
    x_size = x.numel()  # Total number of elements.
    D_per_group = D // group

    result = torch.empty_like(x)
    BLOCK_SIZE = 256
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_fwd_kernel_2d[(num_blocks,)](
        x, n, d, result,
        B, D, H, W,
        group, x_size, D_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return result

# ==============================================================================
# 2D Backward Kernel: computes gradients w.r.t. input and coefficients.
# ==============================================================================

@triton.jit
def rational_bwd_kernel_2d(
    grad_output_ptr, x_ptr, a_ptr, b_ptr,
    d_x_ptr, d_a_ptr, d_b_ptr,
    B: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    group: tl.constexpr, x_size: tl.constexpr, n_size: tl.constexpr, d_size: tl.constexpr,
    D_per_group: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < x_size

    # Load grad_output and input x.
    grad_o = tl.load(grad_output_ptr + idx, mask=mask)
    x_val = tl.load(x_ptr + idx, mask=mask)

    # Compute the D coordinate.
    d_index = (idx % (D * H * W)) // (H * W)
    g_index = d_index // D_per_group
    a_offset = g_index * 6
    b_offset = g_index * 4

    # Load coefficients for the numerator.
    a0 = tl.load(a_ptr + a_offset + 0)
    a1 = tl.load(a_ptr + a_offset + 1)
    a2 = tl.load(a_ptr + a_offset + 2)
    a3 = tl.load(a_ptr + a_offset + 3)
    a4 = tl.load(a_ptr + a_offset + 4)
    a5 = tl.load(a_ptr + a_offset + 5)

    # Load coefficients for the denominator.
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

    # Compute absolute x and its powers.
    axp = tl.abs(x_val)
    axp2 = axp * axp
    axp3 = axp2 * axp
    axp4 = axp3 * axp

    # Evaluate P, Q, R, S.
    P = a0 + a1 * xp + a2 * xp2 + a3 * xp3 + a4 * xp4 + a5 * xp5
    Q = 1.0 + b0_abs * axp + b1_abs * axp2 + b2_abs * axp3 + b3_abs * axp4
    R = a1 + 2.0 * a2 * xp + 3.0 * a3 * xp2 + 4.0 * a4 * xp3 + 5.0 * a5 * xp4

    sign_x = tl.where(x_val < 0, -1.0, 1.0)
    S = sign_x * (b0_abs + 2.0 * b1_abs * axp + 3.0 * b2_abs * axp2 + 4.0 * b3_abs * axp3)

    mpq2 = -P / (Q * Q)
    dx = (R / Q + S * mpq2) * grad_o
    tl.store(d_x_ptr + idx, dx, mask=mask)

    # Compute gradients w.r.t numerator coefficients.
    da0 = grad_o / Q
    da1 = xp * grad_o / Q
    da2 = xp2 * grad_o / Q
    da3 = xp3 * grad_o / Q
    da4 = xp4 * grad_o / Q
    da5 = xp5 * grad_o / Q

    # Compute gradients w.r.t denominator coefficients.
    sign_b0 = tl.where(b0 < 0, -1.0, 1.0)
    sign_b1 = tl.where(b1 < 0, -1.0, 1.0)
    sign_b2 = tl.where(b2 < 0, -1.0, 1.0)
    sign_b3 = tl.where(b3 < 0, -1.0, 1.0)
    db0 = mpq2 * sign_b0 * axp * grad_o
    db1 = mpq2 * sign_b1 * axp2 * grad_o
    db2 = mpq2 * sign_b2 * axp3 * grad_o
    db3 = mpq2 * sign_b3 * axp4 * grad_o

    # Accumulate contributions for numerator coefficients.
    tl.atomic_add(d_a_ptr + (a_offset + 0), da0, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 1), da1, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 2), da2, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 3), da3, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 4), da4, mask=mask)
    tl.atomic_add(d_a_ptr + (a_offset + 5), da5, mask=mask)

    # Accumulate contributions for denominator coefficients.
    tl.atomic_add(d_b_ptr + (b_offset + 0), db0, mask=mask)
    tl.atomic_add(d_b_ptr + (b_offset + 1), db1, mask=mask)
    tl.atomic_add(d_b_ptr + (b_offset + 2), db2, mask=mask)
    tl.atomic_add(d_b_ptr + (b_offset + 3), db3, mask=mask)


def rational_bwd_triton_2d(grad_output: Tensor, x: Tensor, n: Tensor, d: Tensor, group: int):
    """
    2D backward helper.
    Expects x and grad_output of shape [B, D, H, W]. Returns gradients for x, numerator, and denominator.
    """
    B, D, H, W = x.shape
    x_size = x.numel()
    n_size = n.numel()
    d_size = d.numel()
    D_per_group = D // group

    d_x = torch.empty_like(x)
    d_n = torch.zeros_like(n, dtype=torch.float32)
    d_d = torch.zeros_like(d, dtype=torch.float32)

    BLOCK_SIZE = 256
    num_blocks = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    rational_bwd_kernel_2d[(num_blocks,)](
        grad_output, x, n, d,
        d_x, d_n, d_d,
        B, D, H, W,
        group, x_size, n_size, d_size, D_per_group,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return d_x, d_n, d_d

# ==============================================================================
# Autograd Functions for 2D
# ==============================================================================

class RationalTriton2D(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input: Tensor, weight_numerator: Tensor, weight_denominator: Tensor, group: int) -> Tensor:
        """
        2D forward: Expects input of shape [B, D, H, W].
        """
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group
        output = rational_fwd_triton_2d(input, weight_numerator, weight_denominator, group)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output: Tensor):
        input, weight_numerator, weight_denominator = ctx.saved_tensors
        group = ctx.group
        d_input, d_weight_numerator, d_weight_denominator = rational_bwd_triton_2d(
            grad_output, input, weight_numerator, weight_denominator, group
        )
        return d_input, d_weight_numerator, d_weight_denominator, None

