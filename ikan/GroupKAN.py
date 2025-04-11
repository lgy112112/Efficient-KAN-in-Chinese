import sys
import os

# 获取当前脚本所在的目录
# script_dir = os.path.dirname(os.path.abspath(__file__))
# # 更改工作目录到脚本所在目录
# os.chdir(script_dir)
# print("\n\n", os.getcwd())


# sys.path.insert(0, 'ikan/groupkan/rational_kat_cu')
from ikan.kat_1dgroup_triton import KAT_Group
import torch
import torch.nn as nn
from functools import partial
from timm.layers import to_2tuple

class GroupKANLinear(nn.Module):
    """
    基本的 GroupKAN 线性层，实现 KAT_Group -> Linear 的组合
    这是 Kolmogorov-Arnold Network 的基本构建块
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            act_mode="gelu",  # KAT_Group 激活函数模式
            drop=0.,
            use_conv=False,  # 是否使用 Conv2d 代替 Linear
            device=None,
            num_groups=8  # 添加组数参数
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 选择线性层类型 (Linear 或 Conv2d)
        # linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        if use_conv:
            linear_layer = partial(nn.Conv2d, kernel_size=1)
        else:     
            linear_layer = nn.Linear
        
        # KAT_Group 激活层
        self.act = KAT_Group(mode=act_mode, device=device, num_groups=num_groups)
        
        # Dropout 层
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        
        # 线性变换层
        self.linear = linear_layer(in_features, out_features, bias=bias)
    
    def forward(self, x):
        x_origin_dim = x.ndim
        if x_origin_dim == 2:
            x = x.unsqueeze(1)
        # 先激活，再dropout，最后线性变换
        x = self.act(x)
        x = self.drop(x)
        if x_origin_dim == 2:
            x = x.squeeze(1)
        x = self.linear(x)
        return x


class GroupKAN(nn.Module):
    """
    基于 GroupKANLinear 构建的多层 Kolmogorov-Arnold Network
    使用 KAT_Group 作为激活函数的变体
    """
    def __init__(
            self,
            layers_hidden,  # 各层的输入输出特征数列表
            act_mode="gelu",
            drop=0.,
            bias=True,
            use_conv=False,
            device=None,
            num_groups=8
    ):
        """
        初始化 GroupKAN 模型

        参数:
            layers_hidden (list): 网络层的输入和输出特征数列表，例如 [64, 128, 64, 3]
            act_mode (str): 激活函数类型，默认为 "gelu"
            drop (float): Dropout 概率
            bias (bool): 是否使用偏置
            use_conv (bool): 是否使用 Conv2d 代替 Linear
            device (str): 计算设备，默认为 None（自动选择）
            num_groups (int): KAT_Group 中的组数，默认为 8
        """
        super(GroupKAN, self).__init__()
        
        # 验证输入参数
        assert len(layers_hidden) >= 2, "至少需要两个元素来定义一个层（输入和输出特征数）"
        
        # 检查特征数是否都是 num_groups 的倍数
        for features in layers_hidden:
            if features % num_groups != 0:
                print(f"警告: 特征数 {features} 不是组数 {num_groups} 的倍数，可能导致运行错误")
        
        # 初始化网络层
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            in_features = layers_hidden[i]
            out_features = layers_hidden[i + 1]
            
            # 为最后一层使用不同的dropout概率（可选）
            layer_drop = drop
            
            self.layers.append(
                GroupKANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    act_mode=act_mode,
                    drop=layer_drop,
                    use_conv=use_conv,
                    device=device,
                    num_groups=num_groups
                )
            )
    
    def forward(self, x):
        """
        执行模型的前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, in_features]

        返回:
            torch.Tensor: 输出张量，形状为 [batch_size, out_features]
        """
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 测试 GroupKAN 模型
    # 注意：所有特征数必须是 num_groups(默认8) 的倍数
    layers_hidden = [64, 128, 64, 32]  # 定义网络结构
    
    # 创建模型
    model = GroupKAN(
        layers_hidden=layers_hidden,
        act_mode="swish",  # 可以选择不同的激活函数：gelu, swish, identity 等
        drop=0.1,
        device=device
    )
    model = model.to(device)
    
    # 测试前向传播
    batch_size = 16
    x = torch.randn(batch_size, layers_hidden[0])  # 输入张量
    x = x.to(device)
    print(f"x device: {x.device}")
    output = model(x)
    print(f"output device: {output.device}")
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型结构:\n{model}")