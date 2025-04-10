# Efficient-KAN-in-Chinese

本仓库收集并整理了多种基于 Kolmogorov-Arnold 网络（KAN）的高效实现，包括 FourierKAN、ChebyKAN、JacobiKAN、TaylorKAN 和 WaveletKAN 等。这些实现旨在提供对不同类型 KAN 模型的深入理解和便捷使用。为了方便观看、阅读和修改，本人基于 [Efficient-KAN](https://github.com/Blealtan/efficient-kan) 仓库的写法对变种 KAN 进行重构。

---

## 目录

- [Efficient-KAN-in-Chinese](#efficient-kan-in-chinese)
  - [目录](#目录)
  - [重要须知](#重要须知)
    - [2025-04-09](#2025-04-09)
    - [2025-03-25](#2025-03-25)
  - [重大更新](#重大更新)
    - [1. 各类KAN默认参数调整](#1-各类kan默认参数调整)
    - [2. GroupKAN](#2-groupkan)
  - [安装](#安装)
    - [从 PyPI 安装](#从-pypi-安装)
    - [从 GitHub 安装（开发版本）](#从-github-安装开发版本)
    - [从源码安装](#从源码安装)
  - [依赖](#依赖)
  - [使用示例](#使用示例)
  - [简介](#简介)
  - [实现](#实现)
    - [KAN](#kan)
    - [FourierKAN](#fourierkan)
    - [ChebyKAN](#chebykan)
    - [JacobiKAN](#jacobikan)
    - [TaylorKAN](#taylorkan)
    - [WaveletKAN](#waveletkan)
  - [应用实例](#应用实例)
  - [参考资料](#参考资料)
  - [许可证](#许可证)
  - [Star History](#star-history)

---

## 重要须知

### 2025-04-11
建议使用源码安装进行可编辑模式的安装（后文有说明），在此之后，进入 `Efficient-KAN-in-Chinese/ikan/groupkan/rational_kat_cu` 进行 `GroupKAN` 驱动安装 `pip install -e .` ，可使用GPU加速GroupKAN

### 2025-04-09
他妈的这次是真可以了。1.3.0版本可包外调用 `GroupKAN`

### 2025-03-25

~~他妈的那个GroupKAN的仓库依赖一直说某个folder不存在，请大家暂且不要使用 `pip install ikan` 安装，而是使用 `git clone https://github.com/lgy112112/Efficient-KAN-in-Chinese.git` 和 `pip install -e .` 安装。~~

他妈的终于给我修复了路径问题，现在 `ikan==1.2.10` 版本可以舒畅地使用

```python
from ikan.GroupKAN import GroupKAN, GroupKANLinear
```

此外，如果你是windows开发者，恰好没有triton且安装报错，请
`pip install triton-windows`

windows开发无罪！

## 重大更新

### 1. 各类KAN默认参数调整

在深入研究各种KAN变体的性能后，我对默认参数进行了关键调整，主要针对初始化方式和缩放系数：

- **所有KAN**：`scale_base`从1.0降至0.5以下
- **ChebyKAN**：`scale_cheby`从1.0降至0.5
- **FourierKAN**：将 `scale_fourier`从1.0降至0.3
- **JacobiKAN**：将 `scale_jacobi`从1.0降至0.4
- **TaylorKAN**：将 `scale_taylor`从1.0降至0.5
- **WaveletKAN**：将 `scale_wavelet`从1.0降至0.5

通过 `KAN.ipynb`可以测试，在相同迭代次数下**超越传统MLP**。测试结果显示，参数调整后的KAN模型不仅训练速度更快，收敛性更好，而且在拟合复杂函数时的精度也明显提高。

### 2. GroupKAN

本项目新增基于Kolmogorov-Arnold Transformer (KAT)的**GroupKAN**实现，这是KAN的一种高效变体。我在源代码基础上修复了CPU无法训练的bug，并修复了不支持2D tensor的bug：

- [KAT (Kolmogorov-Arnold Transformer)](https://github.com/Adamdad/kat) - 由Xingyi Yang和Xinchao Wang开发，GroupKAN基于此实现
- [rational_kat_cu](https://github.com/Adamdad/rational_kat_cu) - KAT的CUDA/Triton实现，为GroupKAN提供了底层支持
- **实现原理**：使用了KAT_Group作为激活函数，替代了传统KAN中的B样条函数
- **性能优势**：相比原始KAN，GroupKAN具有更快的训练速度和更高的计算效率
- **CUDA支持**：底层使用CUDA/Triton实现的Rational函数，提供了卓越的性能
- **简化结构**：采用"先激活后线性变换"的结构设计，这符合Kolmogorov-Arnold定理的核心思想

你可以使用以下代码创建并测试GroupKAN模型：

```python
from ikan.GroupKAN import GroupKAN

# 定义网络层结构（确保每层特征数是num_groups的倍数）
layers_hidden = [64, 128, 64, 32]

# 创建模型
model = GroupKAN(
    layers_hidden=layers_hidden,
    act_mode="swish",  # 可选: "gelu", "swish", "identity"
    drop=0.1,
    num_groups=8
)

# 使用torchinfo查看模型结构
from torchinfo import summary
summary(model, input_size=(16, 64))
```

---

## 安装

### 从 PyPI 安装

可以直接通过 [PyPI](https://pypi.org/project/efficient-kan/) 使用 `pip` 进行安装：

```bash
pip install ikan
```

随后请不要忘记重启一下IDE以保证安装完整。

### 从 GitHub 安装（开发版本）

若需要安装最新的开发版本，可以从 GitHub 仓库直接安装：

```bash
pip install git+https://github.com/lgy112112/Efficient-KAN-in-Chinese.git
```

随后请不要忘记重启一下IDE以保证安装完整。

### 从源码安装

你也可以从源码安装：

1. 克隆项目仓库：
   ```bash
   git clone https://github.com/lgy112112/Efficient-KAN-in-Chinese.git
   ```
2. 进入项目目录：
   ```bash
   cd Efficient-KAN-in-Chinese
   ```
3. 使用可编辑模式安装：
   ```bash
   pip install -e .
   ```

随后请不要忘记重启一下IDE以保证安装完整。

---

## 依赖

本项目依赖以下 Python 库：

- `torch>=1.9.0`
- `torchinfo`
- `numpy`

通过 `pip` 安装时会自动安装这些依赖。

---

## 使用示例

以下是如何使用本项目的一个简单示例：

```python
from ikan.ChebyKAN import ChebyKAN

model = ChebyKAN(
    layers_hidden=layers_hidden,
    degree=5,
    scale_base=1.0,
    scale_cheby=1.0,
    base_activation=torch.nn.SiLU,
    use_bias=True,
)

summary(model, input_size=(64,))

```

---

## 简介

Kolmogorov-Arnold 网络（KAN）是一类基于 Kolmogorov-Arnold 表示定理的神经网络架构，具有强大的非线性表达能力。本仓库对多种 KAN 的变体进行了实现，包括使用不同基函数（如傅里叶级数、Chebyshev 多项式、Jacobi 多项式、泰勒级数和小波变换）的方法。

## 实现

### KAN

基础的 KAN 实现，使用了 B 样条作为基函数，提供了对 KAN 模型的基本理解。

- 源代码：[KAN.py](ikan\KAN.py)

### FourierKAN

使用傅里叶级数作为基函数的 KAN 实现，能够捕捉输入数据的周期性特征。

- 源代码：[FourierKAN.py](ikan\FourierKAN.py)

### ChebyKAN

使用 Chebyshev 多项式作为基函数的 KAN 实现，具有良好的数值稳定性和逼近能力。

- 源代码：[ChebyKAN.py](ikan\ChebyKAN.py)

### JacobiKAN

使用 Jacobi 多项式作为基函数的 KAN 实现，通过调整参数 \( a \) 和 \( b \)，可以灵活地适应不同的数据分布。

- 源代码：[JacobiKAN.py](ikan\JacobiKAN.py)

### TaylorKAN

使用泰勒级数展开作为基函数的 KAN 实现，适用于需要高阶非线性特征的任务。

- 源代码：[TaylorKAN.py](ikan\TaylorKAN.py)

### WaveletKAN

使用小波变换作为基函数的 KAN 实现，能够捕捉数据的局部特征或频域特征。

- 源代码：[WaveletKAN.py](ikan\WaveletKAN.py)

## 应用实例

以下是本人使用 KAN 进行的项目，欢迎大家复现并探讨：

- **MIMI-MNIST 教程**：在 MNIST 数据集上应用 KAN 模型的教程，展示了如何构建和训练 KAN 来处理手写数字识别任务。

  - 项目地址：[MIMI-MNIST-Tutorial](https://github.com/lgy112112/MIMI-MNIST-Tutorial)
- **股票预测教程**：使用 KAN 模型进行股票价格预测的教程，包括数据预处理、模型构建和结果分析。

  - 项目地址：[Stocks_Prediction_Tutorial](https://github.com/lgy112112/Stocks_Prediction_Tutorial)
  - 项目地址：[KAN_Stocks](https://github.com/lgy112112/KAN_Stocks)
- **KAN 与 VGG 在 CIFAR-10 上的比较**：比较了 KAN 模型和 VGG 网络在 CIFAR-10 数据集上的分类性能，展示了 KAN 的潜力。

  - 项目地址：[KANvsVGGonCIFAR10](https://github.com/lgy112112/KANvsVGGonCIFAR10)

## 参考资料

特别感谢以下开源项目对本仓库的支持和贡献：

- [EfficientKAN](https://github.com/Blealtan/efficient-kan)
- [JacobiKAN](https://github.com/SpaceLearner/JacobiKAN)
- [TaylorKAN](https://github.com/Muyuzhierchengse/TaylorKAN/)
- [Wav-KAN](https://github.com/zavareh1/Wav-KAN)
- [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)
- [FourierKAN](https://github.com/GistNoesis/FourierKAN/)
- [KAT (Kolmogorov-Arnold Transformer)](https://github.com/Adamdad/kat) - 由Xingyi Yang和Xinchao Wang开发，GroupKAN基于此实现
- [rational_kat_cu](https://github.com/Adamdad/rational_kat_cu) - KAT的CUDA/Triton实现，为GroupKAN提供了底层支持

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lgy112112/Efficient-KAN-in-Chinese)](https://star-history.com/#lgy112112/Efficient-KAN-in-Chinese&Date)

---

欢迎大家提出建议和改进，共同完善本仓库。如有任何问题，请提交 [Issue](https://github.com/lgy112112/Efficient-KAN-in-Chinese/issues) 或 [Pull Request](https://github.com/lgy112112/Efficient-KAN-in-Chinese/pulls)。
