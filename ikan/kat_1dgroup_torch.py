import torch
from torch import nn
import os
import json


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


def Rational_CUDA_A_1DGroup(x, weight_numerator, weight_denominator, group):
    """
    Computes the rational function P(x) / Q(x) group-wise where P and Q are polynomials defined by
    the given weights for their coefficients for each group.
    P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
                1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|
    
    Args:
    - x (torch.Tensor): Input tensor of shape (B, L, D).
    - weight_numerator (torch.Tensor): Coefficients of the numerator polynomial for each group.
                                       Shape (group, len_num).
    - weight_denominator (torch.Tensor): Coefficients of the denominator polynomial for each group.
                                         Shape (group, len_deno).
    
    Returns:
    - torch.Tensor: Result of the rational function computation of shape (B, L, D).
    """
    device = x.device
    B, L, D = x.shape
    len_num = weight_numerator.size(1)
    len_deno = weight_denominator.size(1)

    # Group-wise application, ensure D is divisible by the number of groups
    D_per_group = D // group

    # Reshape x to apply each group's parameters separately
    z = x.view(B, L, group, D_per_group).permute(2, 0, 1, 3).contiguous()  # Shape: (group, B, L, D_per_group)
    z = z.view(group, B * L * D_per_group)  # Flatten for group-wise operation

    # Generate powers of z for polynomial terms, assuming _get_xps function supports batched operation
    xps = _get_xps(z, len_num, len_deno)  # Should output shape: (group, B * L * D_per_group, max(len_num, len_deno))

    # Compute numerator as a dot product of powers of z and weights
    numerator = torch.bmm(weight_numerator.unsqueeze(1), xps).squeeze(1)  # Shape: (group, B * L * D_per_group)

    # Compute denominator similarly, considering absolute values
    expanded_dw = torch.cat([
        torch.ones(group, 1, device=device),  # 1 for the constant term of denominator
        weight_denominator,
        torch.zeros(group, max(0, len_num - len_deno - 1), device=device)  # Pad with zeros if numerator degree is higher
    ], dim=1)

    denominator = torch.bmm(expanded_dw.abs().unsqueeze(1), xps).squeeze(1)  # Shape: (group, B * L * D_per_group)

    # Compute the rational function result
    result = numerator.div(denominator)

    # Reshape and reorder to match original x shape
    result = result.view(group, B, L, D_per_group).permute(1, 2, 0, 3).contiguous()  # Shape: (B, L, group, D_per_group)
    result = result.view(B, L, D)  # Shape: (B, L, D)

    return result


class KAT_Group_Torch(nn.Module):
    def __init__(self, num_groups=8, mode="gelu"):
        """
        Initialize the KAT_1DGroup module.

        Args:
            num_groups (int): Number of groups to divide the input for separate processing.
            init_mode (str): Initialization mode which determines the preset weights from JSON file.
        """
        super(KAT_Group_Torch, self).__init__()
        self.order = (5, 4)
        self.num_groups = num_groups
        # Initialize parameters for each group
        self.initialize(mode=mode)
        
    def init_info(self):
        """
        Initialize weights from a JSON file based on the specified mode."""
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
        return data
                
    def initialize(self, mode="gelu"):
        """
        Initialize weights from a JSON file based on the specified mode.

        Args:
            mode (str): The initialization mode.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"])
            weight_numerator = torch.cat([weight_numerator]).view(1, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator]*self.num_groups).view(self.num_groups, -1)
             
            self.weight_numerator = nn.Parameter(torch.FloatTensor(weight_numerator)
                                                      , requires_grad=True) 
            self.weight_denominator = nn.Parameter(torch.FloatTensor(weight_denominator)
                                                      , requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            
    def forward(self, input):
        """
        Forward pass of the module.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Processed tensor.
        """

        assert input.dim() == 3, "Input tensor must be 3D. Of size (batch, length, channels)."
    
        # select the first group, and repeat the weights for all groups
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
        return Rational_CUDA_A_1DGroup(input, weight_numerator, self.weight_denominator, self.num_groups)
        
    
    def extra_repr(self):
        """
        Extra representation of the module for debugging.

        Returns:
            str: String representation of the module's configuration.
        """
        return f'num_groups={self.num_groups}, order={self.order}'
    