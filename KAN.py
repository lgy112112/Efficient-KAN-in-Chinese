import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数

        # 计算网格步长，并生成网格
        #         网格的作用
        # 定义B样条基函数的位置：

        # B样条基函数是在特定的支持点上进行计算的，这些支持点由网格确定。
        # 样条基函数在这些网格点上具有特定的值和形状。
        # 确定样条基函数的间隔：

        # 网格步长（h）决定了网格点之间的距离，从而影响样条基函数的平滑程度和覆盖范围。
        # 网格越密集，样条基函数的分辨率越高，可以更精细地拟合数据。
        # 构建用于插值和拟合的基础：

        # 样条基函数利用这些网格点进行插值，能够构建出连续的、平滑的函数。
        # 通过这些基函数，可以实现输入特征的复杂非线性变换。
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 注册网格作为模型的buffer
        #         在PyTorch中，buffer是一种特殊类型的张量，它在模型中起到辅助作用，但不会作为模型参数进行更新。buffer通常用于存储一些在前向和后向传播过程中需要用到的常量或中间结果。buffer和模型参数一样，会被包含在模型的状态字典中（state dictionary），可以与模型一起保存和加载。

        # register_buffer 的作用
        # self.register_buffer("grid", grid) 的作用是将 grid 注册为模型的一个buffer。这样做有以下几个好处：

        # 持久化：buffer会被包含在模型的状态字典中，可以通过 state_dict 方法保存模型时一并保存，加载模型时也会一并恢复。这对于训练和推理阶段都很有用，确保所有相关的常量都能正确加载。

        # 无需梯度更新：buffer不会在反向传播过程中计算梯度和更新。它们是固定的，只在前向传播中使用。这对于像网格点这样的常量非常适合，因为这些点在训练过程中是固定的，不需要更新。

        # 易于使用：注册为 buffer 的张量可以像模型参数一样方便地访问和使用，而不必担心它们会被优化器错误地更新。

        # 初始化网络参数和超参数

        # 初始化基础权重参数，形状为 (out_features, in_features)
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化样条权重参数，形状为 (out_features, in_features, grid_size + spline_order)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )

        # 如果启用了独立缩放样条功能，初始化样条缩放参数，形状为 (out_features, in_features)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        # 噪声缩放系数，用于初始化样条权重时添加噪声
        self.scale_noise = scale_noise

        # 基础权重的缩放系数，用于初始化基础权重时的缩放因子
        self.scale_base = scale_base

        # 样条权重的缩放系数，用于初始化样条权重时的缩放因子
        self.scale_spline = scale_spline

        # 是否启用独立的样条缩放功能
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

        # 基础激活函数实例，用于对输入进行非线性变换
        self.base_activation = base_activation()

        # 网格更新时的小偏移量，用于在更新网格时引入微小变化，避免过拟合
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # 使用kaiming_uniform_方法初始化基础权重参数base_weight
        # 这个方法基于何凯明初始化，适用于具有ReLU等非线性激活函数的网络
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        
        with torch.no_grad():
            # 为样条权重参数spline_weight添加噪声进行初始化
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            
            # 计算样条权重参数的初始值，结合了scale_spline的缩放因子
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                # 作者此前使用了一般的初始化，效果不佳
                # 使用kaiming_uniform_方法初始化样条缩放参数spline_scaler
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)


    def b_splines(self, x: torch.Tensor):
        """
        计算给定输入张量的B样条基函数。
        B样条（B-splines）是一种用于函数逼近和插值的基函数。
        它们具有局部性、平滑性和数值稳定性等优点，广泛应用于计算机图形学、数据拟合和机器学习中。
        在这段代码中，B样条基函数用于在输入张量上进行非线性变换，以提高模型的表达能力。
        在KAN（Kolmogorov-Arnold Networks）模型中，B样条基函数用于将输入特征映射到高维空间中，以便在该空间中进行线性变换。
        具体来说，B样条基函数能够在给定的网格点上对输入数据进行插值和逼近，从而实现复杂的非线性变换。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: B样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_features

        # 获取网格点（包含在buffer中的self.grid）
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)

        # 为了进行逐元素操作，将输入张量的最后一维扩展一维
        x = x.unsqueeze(-1)

        # 初始化B样条基函数的基矩阵
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        # 迭代计算样条基函数
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # 确保B样条基函数的输出形状正确
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()


    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算插值给定点的曲线的系数。
        curve2coeff 方法用于计算插值给定点的曲线的系数。
        这些系数用于表示插值曲线在特定点的形状和位置。
        具体来说，该方法通过求解线性方程组来找到B样条基函数在给定点上的插值系数。
        此方法的作用是根据输入和输出点计算B样条基函数的系数，
        使得这些基函数能够精确插值给定的输入输出点对。
        这样可以用于拟合数据或在模型中应用非线性变换。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。

        返回:
            torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        # 确保输入张量的维度是2，并且其列数等于输入特征数
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # 确保输出张量的形状正确
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # 计算B样条基函数
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        
        # 转置输出张量
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        
        # 使用线性代数方法求解线性方程组，找到系数
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        
        # 调整结果的形状
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        # 确保结果张量的形状正确
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        
        # 返回连续存储的结果张量
        return result.contiguous()


    @property
    def scaled_spline_weight(self):
        """
        计算带有缩放因子的样条权重。

        样条缩放：如果启用了 enable_standalone_scale_spline，
        则将 spline_scaler 张量扩展一维后与 spline_weight 相乘，
        否则直接返回 spline_weight。

        具体来说，spline_weight 是一个三维张量，形状为 (out_features, in_features, grid_size + spline_order)。
        而 spline_scaler 是一个二维张量，形状为 (out_features, in_features)。
        为了使 spline_scaler 能够与 spline_weight 逐元素相乘，
        需要将 spline_scaler 的最后一维扩展，以匹配 spline_weight 的第三维。

        返回:
            torch.Tensor: 带有缩放因子的样条权重张量。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )


    def forward(self, x: torch.Tensor):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        # 确保输入张量的最后一维大小等于输入特征数
        assert x.size(-1) == self.in_features
        
        # 保存输入张量的原始形状
        original_shape = x.shape
        
        # 将输入张量展平为二维
        x = x.view(-1, self.in_features)

        # 计算基础线性变换的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # 计算B样条基函数的输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        
        # 合并基础输出和样条输出
        output = base_output + spline_output
        
        # 恢复输出张量的形状
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output


    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        update_grid 方法用于根据输入数据动态更新B样条的网格点，从而适应输入数据的分布。
        该方法通过重新计算和调整网格点，确保B样条基函数能够更好地拟合数据。
        这在训练过程中可能会提高模型的精度和稳定性。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            margin (float): 网格更新的边缘大小，用于在更新网格时引入微小变化。
        """
        # 确保输入张量的维度正确
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)  # 获取批量大小

        # 计算输入张量的B样条基函数
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # 转置为 (in, batch, coeff)

        # 获取当前的样条权重
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # 转置为 (in, coeff, out)

        # 计算未缩减的样条输出
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # 转置为 (batch, in, out)

        # 为了收集数据分布，对每个通道分别进行排序
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # 计算均匀步长，并生成均匀网格
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # 混合均匀网格和自适应网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        # 扩展网格以包括样条边界
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        # 更新模型中的网格点
        self.grid.copy_(grid.T)

        # 重新计算样条权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算正则化损失。

        这是对论文中提到的原始L1正则化的一种简单模拟，因为原始方法需要从
        展开的 (batch, in_features, out_features) 中间张量计算绝对值和熵，
        但如果我们想要一个高效的内存实现，这些张量会被隐藏在F.linear函数后面。

        现在的L1正则化计算为样条权重的平均绝对值。
        作者的实现还包括这个项，此外还有基于样本的正则化。
        """
        # 计算样条权重的绝对值的平均值
        l1_fake = self.spline_weight.abs().mean(-1)
        
        # 计算激活正则化损失，即所有样条权重绝对值的和
        regularization_loss_activation = l1_fake.sum()
        
        # 计算每个权重占总和的比例
        p = l1_fake / regularization_loss_activation
        
        # 计算熵正则化损失，即上述比例的负熵
        regularization_loss_entropy = -torch.sum(p * p.log())
        
        # 返回总的正则化损失，包含激活正则化和熵正则化
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


########################################################## 类定义
class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        初始化KAN模型。

        参数:
            layers_hidden (list): 每层的输入和输出特征数列表。
            grid_size (int): 网格大小。
            spline_order (int): 样条阶数。
            scale_noise (float): 样条权重初始化时的噪声缩放系数。
            scale_base (float): 基础权重初始化时的缩放系数。
            scale_spline (float): 样条权重初始化时的缩放系数。
            base_activation (nn.Module): 基础激活函数类。
            grid_eps (float): 网格更新时的小偏移量。
            grid_range (list): 网格范围。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size  # 网格大小
        self.spline_order = spline_order  # 样条阶数

        # 初始化模型层
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        """
        实现模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否在前向传播过程中更新网格。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        计算模型的正则化损失。

        参数:
            regularize_activation (float): 激活正则化系数。
            regularize_entropy (float): 熵正则化系数。

        返回:
            float: 总的正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
