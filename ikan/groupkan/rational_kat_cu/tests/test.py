import torch
import kat_rational
from rational.torch import Rational
from torch import nn

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

class My_rational(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = kat_rational.rational_fwd(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = kat_rational.rational_bwd(grad_output, x, w_numerator, w_denominator)
        return d_x, d_weight_numerator, d_weight_denominator, None

class My_rational_optimized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = kat_rational.rational_fwd_optimized(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = kat_rational.rational_bwd_optimized(grad_output, x, w_numerator, w_denominator)
        return d_x, d_weight_numerator, d_weight_denominator, None

def test_forward(x, numerator_weights, denominator_weights):
    
    print("Testing forward pass")
    # Perform the rational function computation
    result = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)

    my_results = My_rational_optimized.apply(x, numerator_weights, denominator_weights)
    print("result", result)
    print("my_results", my_results)

    # Check if the results match
    assert torch.allclose(result, my_results, atol=1e-6), "Forward pass results do not match"
    print("Forward pass test passed")
    print("#"*50)
    return result

def test_backward(x, numerator_weights, denominator_weights):
    print("Testing backward pass")
    
    expected_output = torch.sigmoid(x)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    act = Rational(approx_func="gelu",).cuda()
    assert torch.allclose(act.numerator, numerator_weights), "Numerator weights do not match"
    assert torch.allclose(act.denominator, denominator_weights), "Denominator weights do not match"

    # Perform the rational function computation
    output = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)
    loss = loss_fn(expected_output, output)
    loss.backward()
    torch_grad_n = numerator_weights.grad.clone()
    torch_grad_d = denominator_weights.grad.clone()
    
    numerator_weights.grad.zero_()
    denominator_weights.grad.zero_()
    
    my_output = My_rational_optimized.apply(x, numerator_weights, denominator_weights)
    loss = loss_fn(expected_output, my_output)
    loss.backward()
    my_grad_n = numerator_weights.grad.clone()
    my_grad_d = denominator_weights.grad.clone()
    
    numerator_weights.grad.zero_()
    denominator_weights.grad.zero_()
    # print(act.numerator)
    # print(act.denominator)
    # act.numerator.grad.zero_()
    # act.denominator.grad.zero_()

    
    off_output = act(x)
    loss = loss_fn(expected_output, off_output)
    loss.backward()
    off_grad_n = act.numerator.grad.clone()
    off_grad_d = act.denominator.grad.clone()
    
    print("my_output", my_output)
    print("off_output", off_output)
    print("torch output", output)
    
    print("my_grad_n", my_grad_n)
    print("torch_grad_n", torch_grad_n)
    print("off_grad_n", off_grad_n)
    print("my_grad_d", my_grad_d)
    print("torch_grad_d", torch_grad_d)
    print("off_grad_d", off_grad_d)


    assert torch.allclose(my_grad_n, off_grad_n), "Numerator gradients do not match"
    assert torch.allclose(my_grad_d, off_grad_d), "Denominator gradients do not match"    
    # Check if the results match
    assert torch.allclose(torch_grad_n, my_grad_n), "Numerator gradients do not match"
    assert torch.allclose(torch_grad_d, my_grad_d), "Denominator gradients do not match"
    
    print("Backward pass test passed")
    print("#"*50)

    # return result
def benchmark_bwd_time(x, numerator_weights, denominator_weights):
    import time
    expected_output = torch.sigmoid(x)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start = time.time()
    for _ in range(100):
        output = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)
        loss = loss_fn(expected_output, output)
        loss.backward()
        torch.cuda.synchronize()
        numerator_weights.grad.detach_()
        numerator_weights.grad.zero_()
        denominator_weights.grad.detach_()
        denominator_weights.grad.zero_()
        
    used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    
    used_time /= 100
    print("Time taken by Rational_CUDA_A_F:", used_time, "Peak memory:", peak_mem)
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start = time.time()
    for _ in range(100):
        my_output = My_rational.apply(x, numerator_weights, denominator_weights)
        loss = loss_fn(expected_output, my_output)
        loss.backward()
        torch.cuda.synchronize()
        
        numerator_weights.grad.detach_()
        numerator_weights.grad.zero_()
        denominator_weights.grad.detach_()
        denominator_weights.grad.zero_()
    used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    used_time /= 100
    print("Time taken by kat_rational.rational_bwd:", used_time, "Peak memory:", peak_mem)
    
    used_time = 0
    torch.cuda.reset_peak_memory_stats()  # Reset peak memory statistics
    start = time.time()
    for _ in range(100):
        my_output = My_rational_optimized.apply(x, numerator_weights, denominator_weights)
        loss = loss_fn(expected_output, my_output)
        loss.backward()
        torch.cuda.synchronize()
        numerator_weights.grad.detach_()
        numerator_weights.grad.zero_()
        denominator_weights.grad.detach_()
        denominator_weights.grad.zero_()
        
    used_time += time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        
    used_time /= 100
    print("Time taken by kat_rational.rational_bwd_optimized:", used_time, "Peak memory:", peak_mem)
    
    

def benchmark_fwd_time(x, numerator_weights, denominator_weights):
    import time
    used_time = 0
    for _ in range(100):
        start = time.time()
        result = Rational_CUDA_A_F(x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    print("Time taken by Rational_CUDA_A_F:", used_time)

    used_time = 0
    for _ in range(100):
        start = time.time()
        my_results = My_rational.apply(x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    print("Time taken by kat_rational.rational_fwd:", used_time)
    
    used_time = 0
    for _ in range(100):
        start = time.time()
        my_results = My_rational_optimized.apply(x, numerator_weights, denominator_weights)
        torch.cuda.synchronize()
        used_time += time.time() - start

    used_time /= 100
    print("Time taken by kat_rational.rational_fwd_optimized:", used_time)
    
    

    return result
if __name__=="__main__":
    # Define tensors for the numerator and denominator coefficients
    numerator_weights = nn.Parameter(torch.tensor([-0.0012423594497499122,
                0.5080497063245629,
                0.41586363182937475,
                0.13022718688035761,
                0.024355900098993424,
                0.00290283948155535], dtype=torch.float32, device='cuda'), requires_grad=True)
    denominator_weights = nn.Parameter(torch.tensor([-0.06675015696494944,
                0.17927646217001553,
                0.03746682605496631,
                1.6561610853276082e-10], dtype=torch.float32, device='cuda'), requires_grad=True)

    # Input tensor
    x = torch.randn(1024, 640, dtype=torch.float32, device='cuda')
    
    
    # test_forward(x, numerator_weights, denominator_weights)

    test_backward(x, numerator_weights, denominator_weights)
    # rat = Rational(cuda=True)
    # expected_output = torch.sigmoid(x)
    # loss_fn = torch.nn.MSELoss(reduction='sum')
    # output = rat(x)
    # loss = loss_fn(expected_output, output)
        
    # loss.backward()
    # print(x.grad)
    # print(rat.numerator.grad)
    # print(rat.denominator.grad)
    # benchmark_bwd_time(x, numerator_weights, denominator_weights)
    # test_forward(x, numerator_weights, denominator_weights)
    # benchmark_fwd_time(x, numerator_weights, denominator_weights)
    