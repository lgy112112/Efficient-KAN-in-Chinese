import torch
import torch.nn.functional as F
import math
from torchinfo import summary

class WaveletKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        wavelet_type='mexican_hat',
        scale_base=1.0,
        scale_wavelet=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 WaveletKANLinear 层。

        参数:
            in_features (int): 输入特征的维度。
            out_features (int): 输出特征的维度。
            wavelet_type (str): 小波类型，可选值有 'mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'。
                该参数指定使用的小波类型，不同的小波具有不同的特性。
            scale_base (float): 基础权重初始化的缩放因子。
                该参数用于在初始化基础权重（即 base_weight）时对初始化值进行缩放。
            scale_wavelet (float): 小波系数初始化的缩放因子。
                该参数控制初始化小波系数（wavelet_weights）时的值范围。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(WaveletKANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.wavelet_type = wavelet_type  # 小波类型
        self.scale_base = scale_base  # 基础权重缩放因子
        self.scale_wavelet = scale_wavelet  # 小波系数缩放因子
        self.base_activation = base_activation()  # 基础激活函数实例
        self.use_bias = use_bias  # 是否使用偏置项

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化小波系数参数，形状为 (out_features, in_features)
        self.wavelet_weights = torch.nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        # 初始化小波的尺度和位移参数
        self.scale = torch.nn.Parameter(torch.ones(out_features, in_features))
        self.translation = torch.nn.Parameter(torch.zeros(out_features, in_features))

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

        # 初始化小波系数参数 wavelet_weights
        with torch.no_grad():
            std = self.scale_wavelet / math.sqrt(self.in_features)
            self.wavelet_weights.uniform_(-std, std)

        # 初始化尺度和位移参数
        torch.nn.init.ones_(self.scale)
        torch.nn.init.zeros_(self.translation)

        if self.use_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.base_weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def wavelet_transform(self, x):
        """
        计算输入 x 的小波变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)

        返回:
            torch.Tensor: 小波变换的输出，形状为 (batch_size, out_features)
        """
        batch_size = x.size(0)

        # 扩展 x 为形状 (batch_size, 1, in_features)
        x_expanded = x.unsqueeze(1)


        # 获取尺度和位移参数，形状为 (out_features, in_features)
        scale = self.scale  # (out_features, in_features)
        translation = self.translation  # (out_features, in_features)

        # 扩展尺度和位移参数，形状为 (1, out_features, in_features)
        scale_expanded = scale.unsqueeze(0)
        translation_expanded = translation.unsqueeze(0)

        # 计算缩放后的输入
        x_scaled = (x_expanded - translation_expanded) / scale_expanded  # (batch_size, out_features, in_features)

        # 选择小波类型并计算小波函数值
        if self.wavelet_type == 'mexican_hat':
            term1 = (x_scaled ** 2 - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # 中心频率
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            # 高斯导数小波（Derivative of Gaussian）
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            # Meyer 小波实现
            pi = math.pi
            v = torch.abs(x_scaled)
            wavelet = torch.sin(pi * v) * self.meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            # Shannon 小波实现
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(
                x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device
            )
            wavelet = sinc * window
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        # 将 wavelet_weights 扩展为形状 (1, out_features, in_features)
        wavelet_weights_expanded = self.wavelet_weights.unsqueeze(0)

        # 计算加权的小波输出
        wavelet_output = (wavelet * wavelet_weights_expanded).sum(dim=2)  # (batch_size, out_features)

        return wavelet_output

    def meyer_aux(self, v):
        """
        Meyer 小波的辅助函数。

        参数:
            v (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 辅助函数的输出
        """
        pi = math.pi

        def nu(t):
            return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

        cond1 = v <= 0.5
        cond2 = (v > 0.5) & (v < 1.0)

        result = torch.zeros_like(v)
        result[cond1] = 1.0
        result[cond2] = torch.cos(pi / 2 * nu(2 * v[cond2] - 1))

        return result

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

        # 计算小波变换的输出
        wavelet_output = self.wavelet_transform(x)  # (batch_size, out_features)

        # 合并基础输出和小波输出
        output = base_output + wavelet_output

        # 加上偏置项
        if self.use_bias:
            output += self.bias

        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)

        return output

    def regularization_loss(self, regularize_coeffs=1.0):
        """
        计算小波系数的正则化损失。

        参数:
            regularize_coeffs (float): 正则化系数。

        返回:
            torch.Tensor: 正则化损失值。
        """
        # 计算小波系数的 L2 范数
        coeffs_l2 = self.wavelet_weights.pow(2).mean()
        return regularize_coeffs * coeffs_l2


class WaveletKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        wavelet_type='mexican_hat',
        scale_base=1.0,
        scale_wavelet=1.0,
        base_activation=torch.nn.SiLU,
        use_bias=True,
    ):
        """
        初始化 WaveletKAN 模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            wavelet_type (str): 小波类型，可选值有 'mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_wavelet (float): 小波系数初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            use_bias (bool): 是否使用偏置项。
        """
        super(WaveletKAN, self).__init__()

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                WaveletKANLinear(
                    in_features,
                    out_features,
                    wavelet_type=wavelet_type,
                    scale_base=scale_base,
                    scale_wavelet=scale_wavelet,
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
    import torch

    # 定义模型的隐藏层结构，每层的输入和输出维度
    layers_hidden = [64, 128, 256, 128, 64, 32]

    # 创建 WaveletKAN 模型
    model = WaveletKAN(
        layers_hidden=layers_hidden,
        wavelet_type='shannon',  # 可选 'mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'
        scale_base=1.0,
        scale_wavelet=1.0,
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
# WaveletKAN                               [32]                      --
# ├─ModuleList: 1-1                        --                        --
# │    └─WaveletKANLinear: 2-1             [128]                     32,896
# │    │    └─SiLU: 3-1                    [1, 64]                   --
# │    └─WaveletKANLinear: 2-2             [256]                     131,328
# │    │    └─SiLU: 3-2                    [1, 128]                  --
# │    └─WaveletKANLinear: 2-3             [128]                     131,200
# │    │    └─SiLU: 3-3                    [1, 256]                  --
# │    └─WaveletKANLinear: 2-4             [64]                      32,832
# │    │    └─SiLU: 3-4                    [1, 128]                  --
# │    └─WaveletKANLinear: 2-5             [32]                      8,224
# │    │    └─SiLU: 3-5                    [1, 64]                   --
# ==========================================================================================
# Total params: 336,480