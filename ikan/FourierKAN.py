import torch
import torch.nn.functional as F
import math
from torchinfo import summary

class FourierKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_frequencies=10,
        scale_base=1.0,
        scale_fourier=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
        smooth_initialization=False,
    ):
        """
        初始化 FourierKANLinear 层。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。

            num_frequencies (int): 傅里叶基频的数量（即频率的数量）。
                这个参数控制傅里叶基函数的数量（正弦和余弦项的频率数量）。
                更高的 num_frequencies 值意味着使用更多的频率，这可以捕捉到输入信号中的更多复杂模式。
                一般情况下，较小的频率对应较为平滑的特征，而较高的频率则可以表示更复杂和快速变化的特征。
                因此，增大 num_frequencies 会增加模型的非线性表达能力，但可能也会导致更复杂的特征学习过程。

            scale_base (float): 基础权重初始化的缩放因子。
                这个参数用于在初始化基础权重（即 base_weight）时对初始化值进行缩放。
                初始化时，模型的权重通常以一定的分布进行采样。
                这个缩放因子可以控制基础线性部分的权重值的大小，从而影响模型在初期的学习能力。
                如果这个值太大，可能会导致梯度爆炸；如果太小，可能会导致梯度消失。
                通过调整这个缩放因子，可以让模型在训练初期更稳定。

            scale_fourier (float): 傅里叶系数初始化的缩放因子。
                这个参数控制初始化傅里叶系数（fourier_coeffs）时的值范围。
                傅里叶系数对应正弦和余弦函数的幅值，较大的 scale_fourier 值会导致这些函数幅值较大，从而增强模型的非线性特征；
                较小的 scale_fourier 值则会减弱模型的非线性特征。
                合理的缩放因子有助于模型在训练初期避免出现过拟合或者欠拟合的情况。

            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。

            smooth_initialization (bool): 是否在初始化时对高频分量进行衰减，以获得平滑的初始函数。
                这个参数控制傅里叶系数的初始化方式。
                如果 smooth_initialization 设置为 True，则高频率的傅里叶分量在初始化时会被衰减，具体表现为频率越高的分量其系数越小。
                这种衰减可以让模型在初期阶段生成更为平滑的函数，避免频率过高的波动，帮助模型在初期训练时更稳定。
                如果该参数为 False，高频分量不会受到衰减，模型在初期可能会生成更多高频特征，
                这可能适合某些更复杂的数据分布，
                但也有可能导致训练初期出现不稳定。
        """
        super(FourierKANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.num_frequencies = num_frequencies  # 傅里叶频率数量
        self.scale_base = scale_base  # 基础权重缩放因子
        self.scale_fourier = scale_fourier  # 傅里叶系数缩放因子
        self.base_activation = base_activation()  # 基础激活函数实例
        self.use_bias = use_bias  # 是否使用偏置项
        self.smooth_initialization = smooth_initialization  # 是否进行平滑初始化

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化傅里叶系数参数，形状为 (2, out_features, in_features, num_frequencies)
        # 2 表示正弦和余弦两个部分
        self.fourier_coeffs = torch.nn.Parameter(
            torch.Tensor(2, out_features, in_features, num_frequencies)
        )

        if self.use_bias:
            # 初始化偏置项，形状为 (out_features,)
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Kaiming 初始化基础权重参数 base_weight
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 初始化傅里叶系数参数 fourier_coeffs
        # 根据是否进行平滑初始化来调整系数的标准差
        with torch.no_grad():
            if self.smooth_initialization:
                # 频率衰减因子，频率越高，初始化的值越小
                frequency_decay = (torch.arange(self.num_frequencies, device=self.fourier_coeffs.device) + 1.0) ** -2.0
            else:
                frequency_decay = torch.ones(self.num_frequencies, device=self.fourier_coeffs.device)
            
            # 计算标准差
            std = self.scale_fourier / math.sqrt(self.in_features) / frequency_decay  # 形状为 (num_frequencies,)
            std = std.view(1, 1, -1)  # 调整形状以便广播

            # 初始化余弦系数
            self.fourier_coeffs[0].uniform_(-1, 1)
            self.fourier_coeffs[0].mul_(std)

            # 初始化正弦系数
            self.fourier_coeffs[1].uniform_(-1, 1)
            self.fourier_coeffs[1].mul_(std)


        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

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
        # 将输入张量展平成 (..., in_features)
        x = x.view(-1, self.in_features)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 计算傅里叶基函数的输出
        # 获取频率 k，形状为 (1, 1, num_frequencies)
        k = torch.arange(1, self.num_frequencies + 1, device=x.device).view(1, 1, -1)

        # 扩展输入 x，形状为 (batch_size, in_features, 1)
        x_expanded = x.unsqueeze(-1)

        # 计算 x 与频率的乘积
        xk = x_expanded * k  # 形状为 (batch_size, in_features, num_frequencies)

        # 计算正弦和余弦部分
        cos_xk = torch.cos(xk)  # 形状为 (batch_size, in_features, num_frequencies)
        sin_xk = torch.sin(xk)

        # 计算傅里叶变换的输出
        # 余弦部分
        cos_part = torch.einsum(
            "bif, oif->bo",
            cos_xk,
            self.fourier_coeffs[0],
        )
        # 正弦部分
        sin_part = torch.einsum(
            "bif, oif->bo",
            sin_xk,
            self.fourier_coeffs[1],
        )

        fourier_output = cos_part + sin_part

        # 合并基础输出和傅里叶输出
        output = base_output + fourier_output

        # 加上偏置项
        if self.use_bias:
            output += self.bias

        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算傅里叶系数的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            torch.Tensor: 正则化损失值。
        """
        # 计算傅里叶系数的 L2 范数
        coeffs_l2 = self.fourier_coeffs.pow(2).mean()
        return regularize_coeffs * coeffs_l2


########################################################## 类定义
class FourierKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        num_frequencies=5,
        scale_base=0.1,
        scale_fourier=0.05,
        base_activation=torch.nn.SiLU,
        use_bias=True,
        smooth_initialization=False,
    ):
        """
        初始化 FourierKAN 模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            num_frequencies (int): 傅里叶基频的数量（即频率的数量）。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_fourier (float): 傅里叶系数初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
            smooth_initialization (bool): 是否在初始化时对高频分量进行衰减，以获得平滑的初始函数。
        """
        super(FourierKAN, self).__init__()

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                FourierKANLinear(
                    in_features,
                    out_features,
                    num_frequencies=num_frequencies,
                    scale_base=scale_base,
                    scale_fourier=scale_fourier,
                    base_activation=base_activation,
                    use_bias=use_bias,
                    smooth_initialization=smooth_initialization,
                )
            )

    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
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

    # 创建5层的FourierKAN模型
    model = FourierKAN(
        layers_hidden=layers_hidden,
        num_frequencies=20,  # 设置傅里叶频率的数量
        scale_base=1.0,
        scale_fourier=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
        smooth_initialization=True
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
# FourierKAN                               [32]                      --
# ├─ModuleList: 1-1                        --                        --
# │    └─FourierKANLinear: 2-1             [128]                     336,000
# │    │    └─SiLU: 3-1                    [1, 64]                   --
# │    └─FourierKANLinear: 2-2             [256]                     1,343,744
# │    │    └─SiLU: 3-2                    [1, 128]                  --
# │    └─FourierKANLinear: 2-3             [128]                     1,343,616
# │    │    └─SiLU: 3-3                    [1, 256]                  --
# │    └─FourierKANLinear: 2-4             [64]                      335,936
# │    │    └─SiLU: 3-4                    [1, 128]                  --
# │    └─FourierKANLinear: 2-5             [32]                      84,000
# │    │    └─SiLU: 3-5                    [1, 64]                   --
# ==========================================================================================
# Total params: 3,443,296