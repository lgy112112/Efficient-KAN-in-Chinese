# Efficient-KAN-in-Chinese

本仓库收集并整理了多种基于 Kolmogorov-Arnold 网络（KAN）的高效实现，包括 FourierKAN、ChebyKAN、JacobiKAN、TaylorKAN 和 WaveletKAN 等。这些实现旨在提供对不同类型 KAN 模型的深入理解和便捷使用。为了方便观看、阅读和修改，本人基于 [Efficient-KAN](https://github.com/Blealtan/efficient-kan) 仓库的写法对变种 KAN 进行重构。

## 目录

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

## 简介

Kolmogorov-Arnold 网络（KAN）是一类基于 Kolmogorov-Arnold 表示定理的神经网络架构，具有强大的非线性表达能力。本仓库对多种 KAN 的变体进行了实现，包括使用不同基函数（如傅里叶级数、Chebyshev 多项式、Jacobi 多项式、泰勒级数和小波变换）的方法。

## 实现

### KAN

基础的 KAN 实现，使用了 B 样条作为基函数，提供了对 KAN 模型的基本理解。

- 源代码：[KAN.py](KAN.py)

### FourierKAN

使用傅里叶级数作为基函数的 KAN 实现，能够捕捉输入数据的周期性特征。

- 源代码：[FourierKAN.py](FourierKAN.py)

### ChebyKAN

使用 Chebyshev 多项式作为基函数的 KAN 实现，具有良好的数值稳定性和逼近能力。

- 源代码：[ChebyKAN.py](ChebyKAN.py)

### JacobiKAN

使用 Jacobi 多项式作为基函数的 KAN 实现，通过调整参数 \( a \) 和 \( b \)，可以灵活地适应不同的数据分布。

- 源代码：[JacobiKAN.py](JacobiKAN.py)

### TaylorKAN

使用泰勒级数展开作为基函数的 KAN 实现，适用于需要高阶非线性特征的任务。

- 源代码：[TaylorKAN.py](TaylorKAN.py)

### WaveletKAN

使用小波变换作为基函数的 KAN 实现，能够捕捉数据的局部特征或频域特征。

- 源代码：[WaveletKAN.py](WaveletKAN.py)

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

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lgy112112/Efficient-KAN-in-Chinese)](https://star-history.com/#lgy112112/Efficient-KAN-in-Chinese&Date)

---

欢迎大家提出建议和改进，共同完善本仓库。如有任何问题，请提交 [Issue](https://github.com/lgy112112/Efficient-KAN-in-Chinese/issues) 或 [Pull Request](https://github.com/lgy112112/Efficient-KAN-in-Chinese/pulls)。