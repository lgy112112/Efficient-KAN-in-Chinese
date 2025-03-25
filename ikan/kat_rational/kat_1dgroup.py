import torch
import kat_rational_cu
from torch import nn
import os
import json
from .kat_1dgroup_torch import Rational_CUDA_A_1DGroup

class rational_1dgroup(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, weight_numerator, weight_denominator, group):
        """
        Forward pass of the custom autograd function.
        
        Args:
            ctx: Context object used to stash information for backward computation.
            input (Tensor): Input tensor.
            weight_numerator (Tensor): Weights of the numerator polynomial.
            weight_denominator (Tensor): Weights of the denominator polynomial.
            group (int): The group number.

        Returns:
            Tensor: The result of the rational function applied to the input tensor.
        """
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        ctx.group = group
        x = kat_rational_cu.rational_fwd_1dgroup(input, weight_numerator, weight_denominator, group)
        return x

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass of the custom autograd function.
        
        Args:
            ctx: Context object from the forward pass.
            grad_output (Tensor): Gradient of the output tensor.

        Returns:
            tuple: Gradients of the input, weight_numerator, weight_denominator.
        """
        input, weight_numerator, weight_denominator = ctx.saved_tensors
        group = ctx.group
        d_input, d_weight_numerator, d_weight_denominator = kat_rational_cu.rational_bwd_1dgroup(grad_output, input, weight_numerator, weight_denominator, group)
        return d_input, d_weight_numerator, d_weight_denominator, None


class KAT_Group(nn.Module):
    def __init__(self, num_groups=8, mode="gelu", device="cuda"):
        """
        Initialize the KAT_Group module.

        Args:
            num_groups (int): Number of groups for separate processing of input.
            mode (str): Initialization mode, determines weights preset from JSON file.
            device (str): Device to run the module on ('cuda' or 'cpu').
        """
        super(KAT_Group, self).__init__()
        assert device in ["cuda", "cpu"], "Device must be either 'cuda' or 'cpu'."
        
        self.order = (5, 4)
        self.num_groups = num_groups

        # Initialize weights based on the given mode
        self.initialize(mode=mode)
        
        # Set the appropriate rational function based on the device
        if device == "cuda":
            self.rational = rational_1dgroup.apply
        else:
            self.rational = Rational_CUDA_A_1DGroup
            
    def init_info(self):
        """
        Load weight initialization information from a JSON file.

        Returns:
            dict: Data loaded from the JSON file.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        with open(f'{cfd}/init.json') as json_file:
            data = json.load(json_file)
        return data
                
    def initialize(self, mode="gelu"):
        """
        Initialize weights from a JSON file based on the specified mode.

        Args:
            mode (str): The initialization mode to use.
        """
        cfd = os.path.dirname(os.path.realpath(__file__))
        try:
            with open(f'{cfd}/init.json') as json_file:
                data = json.load(json_file)

            # Extract weights from the JSON data
            weight_numerator = torch.tensor(data[mode]["init_w_numerator"]).view(1, -1)
            weight_denominator = torch.tensor(data[mode]["init_w_denominator"])
            weight_denominator = torch.cat([weight_denominator] * self.num_groups).view(self.num_groups, -1)
             
            # Register weights as trainable parameters
            self.weight_numerator = nn.Parameter(weight_numerator.float(), requires_grad=True)
            self.weight_denominator = nn.Parameter(weight_denominator.float(), requires_grad=True) 

        except FileNotFoundError:
            print("Initialization JSON file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON.")
            
    def forward(self, input):
        """
        Forward pass of the module.

        Args:
            input (Tensor): 3D or 2D input tensor.

        Returns:
            Tensor: Processed tensor after applying rational function.
        """
        assert input.dim() == 3 or input.dim() == 2, "Input tensor must be 3D (batch, length, channels) or 2D (batch, channels)."
    
        # Repeat the weights for all groups
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
        return self.rational(input, weight_numerator, self.weight_denominator, self.num_groups)
        
    def extra_repr(self):
        """
        Extra representation of the module for debugging.

        Returns:
            str: String representation of the module's configuration.
        """
        return f'num_groups={self.num_groups}, order={self.order}'
    
    
if __name__=="__main__":
    
    model = KAT_1DGroup()
    x = torch.linspace(-2, 2, 100)
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    x = x.unsqueeze(0).unsqueeze(0)
    y = model(x.cuda())
    x = x.squeeze(0).squeeze(0)
    y = y.squeeze(0).squeeze(0)
    # plot y vs x
    import matplotlib.pyplot as plt
    plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy())
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.title("Response of KAT_1DGroup")
    plt.grid(True)
    plt.savefig("kat_1dgroup.png")
    
    