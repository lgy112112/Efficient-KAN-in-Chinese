# CUDA/Triton Rational Function for Kolmogorov–Arnold Transformer (KAT)

This CUDA C++ extension facilitates the use of group rational functions in Kolmogorov–Arnold Transformers (KAT). It support the training and inference of https://github.com/Adamdad/kat.

# News
- [x] 2025.2.1 The `triton` version of GR-KAT has been implemented. Installing and running KAT is now much easier!
- [x] 2025.2.2 We implement the 2D version of GR-KAN, using `triton`. The input tensor size is now support `[B, C, H, W]`.

# Installation 
To install the extension, follow these steps:
```shell
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
pip install -e .
```

# Usage
Incorporate the module into your neural network models as shown in the example below, which uses the rational function as an activation layer in a simple two-layer KAN architecture.
```python
from kat_rational import KAT_Group
class KAN(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type="KAT", act_init=["identity", "gelu"]),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode = act_cfg['act_init'][0])
        self.drop1 = nn.Dropout(drop)
        self.act2 = KAT_Group(mode = act_cfg['act_init'][1])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x
```

Note: 
1. For `[B, L, C]` and `[B, C]` input, please use `KAT_Group` class, which support tensors where channels comes in the end.
2. For `[B, C, H, W]` input, please use `KAT_Group2D`. 

PPS: I'm not a CUDA expert 😅. If you run into any issues or have suggestions for the code, please feel free to reach out or submit a pull request! 🚀

# Add new function 

To add new functions to the module:
1. Open `kat_rational/fit.py`.
2. Implement your custom function within this file.
3. Add your function to `fit_and_plot_activation` to evaluate and visualize its performance.

# Example
- Run GR-KAN on MNIST
```shell
python example/mnist.py
```
Results
```shell
# Baseline (GELU Activation)
GELU - Epoch 1: Loss 0.4548 | Epoch 10: Loss 0.0623
Training Time: 84.52 seconds | Test Accuracy: 97.46%

# Optimized (KAT 1DGroup Rational Activation)
KAT 1DGroup - Epoch 1: Loss 0.3401 | Epoch 10: Loss 0.0245
Training Time: 89.11 seconds | Test Accuracy: 97.53%
```

- Run GR-KAN-2D on CIFAR10
```shell
python example/cifar10.py
```
Results
```shell
ReLU Training completed in 136.78 seconds.
ReLU Testing Accuracy: 76.60%, Total time: 138.47 seconds.

KAT 2DGroup Training completed in 416.74 seconds.
KAT 2DGroup Testing Accuracy: 80.08%, Total time: 418.46 seconds.
```

# Acknowlegement

We extend our gratitude to the [rational_activations](https://github.com/ml-research/rational_activations) project for providing the foundational CUDA implementation of rational functions.