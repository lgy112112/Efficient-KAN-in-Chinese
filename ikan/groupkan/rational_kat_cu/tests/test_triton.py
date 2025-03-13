import torch
from kat_rational.rational_triton import RationalTriton1DGroup, rational_fwd_triton
import torch.nn as nn

def _get_xps(z, len_numerator, len_denominator):
    """
    Generates a tensor of powers of the input tensor `z` up to the maximum order 
    needed for the numerator or denominator, whichever is higher.
    
    Args:
    - z (torch.Tensor): The input tensor for which powers are computed.
    - len_numerator (int): Degree of the numerator polynomial plus one.
    - len_denominator (int): Degree of the denominator polynomial plus one.
    
    Returns:
    - torch.Tensor: Tensor where each row contains powers of `z` from 0 to max degree.
    """
    xps = [z]
    for _ in range(max(len_numerator, len_denominator) - 2):
        xps.append(xps[-1] * z)
    xps.insert(0, torch.ones_like(z))  # Add x^0 = 1
    return torch.stack(xps, dim=1)

def Rational_CUDA_A_F(x, weight_numerator, weight_denominator):
    """
    Computes the rational function P(x) / Q(x) where P and Q are polynomials defined by
    the given weights for their coefficients.
    P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
                1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|
    
    Args:
    - x (torch.Tensor): Input tensor.
    - weight_numerator (torch.Tensor): Coefficients of the numerator polynomial.
    - weight_denominator (torch.Tensor): Coefficients of the denominator polynomial.
    
    Returns:
    - torch.Tensor: Result of the rational function computation.
    """
    device = weight_numerator.device
    z = x.view(-1)  # Flatten x to a 1D tensor
    len_num, len_deno = len(weight_numerator), len(weight_denominator)

    # Generate powers of z for polynomial terms
    xps = _get_xps(z, len_num, len_deno)

    # Compute the numerator as a dot product of powers of z and weights
    numerator = (xps * weight_numerator).sum(dim=1)

    # Prepare denominator weights with zero-padding as necessary
    expanded_dw = torch.cat([
        torch.tensor([1.]).to(device),  # 1 for the constant term of denominator
        weight_denominator,
        torch.zeros(max(0, len_num - len_deno - 1)).to(device)  # Pad with zeros if numerator degree is higher
    ])

    # Compute the denominator similarly, considering absolute values
    denominator = (xps * expanded_dw).abs().sum(dim=1)

    return numerator.div(denominator).view(x.shape)  # Reshape result to match input shape

def process_groups(B, L, D, group, x, weights_numerator, weights_denominator):
    """
    Applies Rational_CUDA_A_F group-wise to an input tensor of shape (B, L, D).
    
    Args:
    - B, L, D (int): Dimensions of the input tensor.
    - group (int): Number of groups.
    - x (torch.Tensor): Input tensor of shape (B, L, D).
    - weights_numerator (list of torch.Tensor): List of tensors, each containing numerator coefficients for a group.
    - weights_denominator (list of torch.Tensor): List of tensors, each containing denominator coefficients for a group.
    
    Returns:
    - torch.Tensor: The result tensor of shape (B, L, D).
    """
    
    D_per_group = D // group
    results = []

    for g in range(group):
        # Slice the input tensor for the current group
        start_idx = g * D_per_group
        end_idx = start_idx + D_per_group
        x_group = x[:, :, start_idx:end_idx]
        x_group = x_group.contiguous()

        # Compute the rational function for the current group
        result_group = Rational_CUDA_A_F(x_group, weights_numerator[g], weights_denominator[g])
        results.append(result_group)

    # Concatenate the results along the depth dimension
    return torch.cat(results, dim=2)

def test():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    # Define dimensions and groups.
    B, L, D = 2, 4, 8  # Example: batch=2, length=4, features=8
    group = 4        # Make sure D is divisible by group
    assert D % group == 0, "D must be divisible by group"

    # Coefficient shapes.
    len_num = 6     # numerator coefficients (degree 5)
    len_deno = 4    # denominator coefficients (degree 3, plus constant)

    # Create random input and coefficients.
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    weight_numerator = torch.randn(group, len_num, device=device, dtype=dtype)
    # weight_numerator_g = weight_numerator.unsqueeze(0).repeat(group, 1)
    weight_denominator = torch.randn(group, len_deno, device=device, dtype=dtype)
    # weight_denominator_g = weight_denominator.unsqueeze(0).repeat(group, 1)

    # Compute outputs.
    y_triton = rational_fwd_triton(x, weight_numerator, weight_denominator, group)
    y_torch  = process_groups(B, L, D, group, x, weight_numerator, weight_denominator)
    
    print(y_torch.dtype, y_triton.dtype)

    # Compute maximum absolute difference.
    diff = (y_triton - y_torch).abs().max().item()
    print("Maximum absolute difference between Triton and Torch implementations:", diff)

    # You can also assert closeness.
    if torch.allclose(y_triton, y_torch, atol=1e-5):
        print("SUCCESS: Triton and Torch implementations produce similar results.")
    else:
        print("WARNING: There is a difference between the implementations.")
    
    # Optionally, print the outputs.
    print("\nTriton output:")
    print(y_triton)
    print("\nTorch output:")
    print(y_torch)
    
    
def test_time():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # Define dimensions and groups.
    B, L, D = 100, 100, 512  # Example: batch=2, length=4, features=8
    group = 4        # Make sure D is divisible by group
    assert D % group == 0, "D must be divisible by group"

    # Coefficient shapes.
    len_num = 6     # numerator coefficients (degree 5)
    len_deno = 4    # denominator coefficients (degree 3, plus constant)


    # Create random input and coefficients.
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    weight_numerator = torch.randn(group, len_num, device=device, dtype=dtype)
    # weight_numerator_g = weight_numerator.unsqueeze(0).repeat(group, 1)
    weight_denominator = torch.randn(group, len_deno, device=device, dtype=dtype)
    # weight_denominator_g = weight_denominator.unsqueeze(0).repeat(group, 1)
    # Benchmark each implementation.
    triton_time, triton_mem = benchmark_with_memory(rational_fwd_triton, x, weight_numerator, weight_denominator, group)
    torch_time, torch_mem = benchmark_with_memory(process_groups, B, L, D, group, x, weight_numerator, weight_denominator)

    print("\nAverage execution time over iterations:")
    print(f"  Triton function: {triton_time:.3f} ms per iteration, {triton_mem:.3f} MB")
    print(f"  Torch function: {torch_time:.3f} ms per iteration, {torch_mem:.3f} MB")
    
def benchmark_with_memory(func, *args, n_iter=100, warmup=10, device=torch.device("cuda")):
    """
    Benchmark a function running on the GPU using CUDA events and report memory usage.

    Returns:
      - avg_time_ms: average execution time in milliseconds.
      - memory_used: maximum memory used (in MB) during the benchmark.
    """
    # Warm-up
    for _ in range(warmup):
        func(*args)
    
    torch.cuda.synchronize(device)

    # Reset the max memory counter.
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iter):
        func(*args)
    end_event.record()

    torch.cuda.synchronize(device)

    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / n_iter

    # Get the peak memory usage in MB.
    max_memory_bytes = torch.cuda.max_memory_allocated(device)
    memory_used = max_memory_bytes / (1024 ** 2)

    return avg_time_ms, memory_used

def test_backward():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    # Define dimensions and groups.
    B, L, D = 2, 4, 8  # Example: batch=2, length=4, features=8
    group = 4        # Make sure D is divisible by group
    assert D % group == 0, "D must be divisible by group"

    # Coefficient shapes.
    len_num = 6     # numerator coefficients (degree 5)
    len_deno = 4    # denominator coefficients (degree 3, plus constant)

    # Create random input and coefficients.
    x = torch.randn(B, L, D, device=device, dtype=dtype)
    weight_numerator = torch.randn(group, len_num, device=device, dtype=dtype)
    weight_numerator = nn.Parameter(weight_numerator, requires_grad=True)
    # weight_numerator_g = weight_numerator.unsqueeze(0).repeat(group, 1)
    weight_denominator = torch.randn(group, len_deno, device=device, dtype=dtype)
    weight_denominator = nn.Parameter(weight_denominator, requires_grad=True)
    
    expected_output = torch.relu(x)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    # weight_denominator_g = weight_denominator.unsqueeze(0).repeat(group, 1)
    # Perform the rational function computation
    output = process_groups(B, L, D, group, x, weight_numerator, weight_denominator)
    loss = loss_fn(expected_output, output)
    loss.backward()
    torch_grad_n = weight_numerator.grad.clone()
    torch_grad_d = weight_denominator.grad.clone()
    
    weight_numerator.grad.zero_()
    weight_denominator.grad.zero_()
    
    my_output = RationalTriton1DGroup.apply(x, weight_numerator, weight_denominator, group)
    loss = loss_fn(expected_output, my_output)
    loss.backward()
    my_grad_n = weight_numerator.grad.clone()
    my_grad_d = weight_denominator.grad.clone()
    
    print(my_output)
    print(output)
    assert torch.allclose(my_output, output, atol=1e-6), "Output mismatch"
    assert torch.allclose(torch_grad_n, my_grad_n), "Numerator gradient mismatch"
    assert torch.allclose(torch_grad_d, my_grad_d), "Denominator gradient mismatch"
    
    print("Backward pass test passed")
# ======================================
# Testing both implementations for equality
# ======================================
if __name__ == "__main__":
    test_time()
    test