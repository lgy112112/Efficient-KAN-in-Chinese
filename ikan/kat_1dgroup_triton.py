
import torch
import torch.nn as nn
from .rational_triton import RationalTriton1DGroup
from .kat_1dgroup_torch import Rational_CUDA_A_1DGroup
import json
import os
# print("\n\nikan/kat_1dgroup_triton.py --- activated\n\n")
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
            self.rational = RationalTriton1DGroup.apply
            # print("\n\nUsing Triton when cuda\n\n")
        else:
            self.rational = Rational_CUDA_A_1DGroup
            # print("\n\nUsing CUDA_A when cpu\n\n")
            
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
    