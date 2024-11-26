import torch
import torch.nn.functional as F
import math
from torchinfo import summary

class TaylorKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        order=3,
        scale_base=1.0,
        scale_taylor=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 TaylorKANLinear 层。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            order (int): 泰勒级数的阶数。
                该参数控制泰勒级数展开的最高阶数，决定了非线性表达的复杂度。
            scale_base (float): 基础权重初始化的缩放因子。
                该参数用于在初始化基础权重（即 base_weight）时对初始化值进行缩放。
            scale_taylor (float): 泰勒系数初始化的缩放因子。
                该参数控制初始化泰勒系数（taylor_coeffs）时的值范围。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(TaylorKANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.order = order  # 泰勒级数的阶数
        self.scale_base = scale_base  # 基础权重缩放因子
        self.scale_taylor = scale_taylor  # 泰勒系数缩放因子
        self.base_activation = base_activation()  # 基础激活函数实例
        self.use_bias = use_bias  # 是否使用偏置项

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化泰勒系数参数，形状为 (out_features, in_features, order)
        self.taylor_coeffs = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, order)
        )

        if self.use_bias:
            # 初始化偏置项，形状为 (out_features,)
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化基础权重参数 base_weight
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )

        # 初始化泰勒系数参数 taylor_coeffs
        with torch.no_grad():
            std = self.scale_taylor / (self.in_features * math.sqrt(self.order))
            self.taylor_coeffs.normal_(mean=0.0, std=std)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def taylor_series(self, x: torch.Tensor):
        """
        计算输入 x 的泰勒级数展开。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)

        返回:
            torch.Tensor: 泰勒级数的输出，形状为 (batch_size, out_features)
        """
        batch_size = x.size(0)

        # 扩展 x，为了与 taylor_coeffs 相乘
        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # (batch_size, 1, in_features, 1)

        # 计算每个阶数的 x 的幂次
        powers = torch.arange(self.order, device=x.device).view(1, 1, 1, -1)  # (1, 1, 1, order)
        x_powers = x_expanded ** powers  # (batch_size, 1, in_features, order)

        # 扩展 taylor_coeffs，以便与 x_powers 相乘
        taylor_coeffs_expanded = self.taylor_coeffs.unsqueeze(0)  # (1, out_features, in_features, order)

        # 计算泰勒级数展开的各项
        taylor_terms = x_powers * taylor_coeffs_expanded  # (batch_size, out_features, in_features, order)

        # 对输入特征维度和阶数维度求和
        taylor_output = taylor_terms.sum(dim=3).sum(dim=2)  # (batch_size, out_features)

        return taylor_output

    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """
        # 保存输入张量的原始形状
        original_shape = x.shape

        # 将输入展平成二维张量，形状为 (-1, in_features)
        x = x.view(-1, self.in_features)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 计算泰勒级数的输出
        taylor_output = self.taylor_series(x)  # (batch_size, out_features)

        # 合并基础输出和泰勒输出
        output = base_output + taylor_output

        # 加上偏置项
        if self.use_bias:
            output += self.bias

        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算泰勒系数的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            torch.Tensor: 正则化损失值。
        """
        # 计算泰勒系数的 L2 范数
        coeffs_l2 = self.taylor_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class TaylorKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        order=3,
        scale_base=1.0,
        scale_taylor=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 TaylorKAN 模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            order (int): 泰勒级数的阶数。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_taylor (float): 泰勒系数初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(TaylorKAN, self).__init__()

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                TaylorKANLinear(
                    in_features,
                    out_features,
                    order=order,
                    scale_base=scale_base,
                    scale_taylor=scale_taylor,
                    base_activation=base_activation,
                    use_bias=use_bias,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (..., in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (..., out_features)。
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_coeffs)
            for layer in self.layers
        )


def demo():

    # 定义模型的隐藏层结构，每层的输入和输出维度
    layers_hidden = [64, 128, 256, 128, 64, 32]

    # 创建 TaylorKAN 模型
    model = TaylorKAN(
        layers_hidden=layers_hidden,
        order=5,  # 设置泰勒级数的阶数
        scale_base=1.0,
        scale_taylor=1.0,
        base_activation=torch.nn.SiLU,  # 使用 SiLU 作为激活函数
        use_bias=True,
    )

    # 打印模型结构
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 使用torchsummary输出模型结构
    summary(model, input_size=(64,))  # 假设输入特征为64维

if __name__ == "__main__":
    demo()

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# TaylorKAN                                [32]                      --
# ├─ModuleList: 1-1                        --                        --
# │    └─TaylorKANLinear: 2-1              [128]                     49,280
# │    │    └─SiLU: 3-1                    [1, 64]                   --
# │    └─TaylorKANLinear: 2-2              [256]                     196,864
# │    │    └─SiLU: 3-2                    [1, 128]                  --
# │    └─TaylorKANLinear: 2-3              [128]                     196,736
# │    │    └─SiLU: 3-3                    [1, 256]                  --
# │    └─TaylorKANLinear: 2-4              [64]                      49,216
# │    │    └─SiLU: 3-4                    [1, 128]                  --
# │    └─TaylorKANLinear: 2-5              [32]                      12,320
# │    │    └─SiLU: 3-5                    [1, 64]                   --
# ==========================================================================================
# Total params: 504,416