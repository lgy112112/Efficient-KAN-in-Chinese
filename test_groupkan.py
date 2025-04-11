from ikan.GroupKAN import GroupKANLinear
from ikan.TaylorKAN import TaylorKANLinear
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
import os


print(f"当前工作目录: {os.getcwd()}")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def train(model, train_loader, test_loader, criterion, optimizer, epochs=100, profile_enabled=False):
    model.to(device)
    model.train()
    best_r2 = -float('inf')
    model_name = model.__class__.__name__
    
    # 创建profiler相关设置
    profiler_schedule = torch.profiler.schedule(
        wait=1,      # 跳过首次迭代
        warmup=1,    # 预热1次迭代
        active=3,    # 实际分析3次迭代
        repeat=1     # 重复整个过程1次
    )
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        
        # 仅在指定epoch启用profiler
        use_profiler = profile_enabled and epoch == 0
        
        if use_profiler:
            # 使用profiler分析训练循环
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile_logs/{model_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    with record_function("train_batch"):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # 更新profiler状态
                    prof.step()
                    
                    # 分析5个batch后停止
                    if batch_idx >= 5:
                        break
                
                # 打印profiler结果
                print("Profiler结果:")
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
                print("\nGPU操作耗时排序:")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else:
            # 普通训练循环
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        
        # 评估阶段
        model.eval()
        test_loss = 0
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, targets).item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # 计算R2分数
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
        ss_res = ((all_targets - all_outputs) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot)
        
        # 保存最佳模型
        if r2 > best_r2:
            best_r2 = r2
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'r2_score': r2,
            }, f'best_{model_name}_model.pth')
        
        # 打印训练和验证损失以及R2分数
        if (epoch+1) % 10 == 0 or epoch == 0 or epoch == epochs-1:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Test Loss: {test_loss/len(test_loader):.4f}, '
                  f'Test R2: {r2:.4f}')

# 添加函数：使用profiler分析单个前向传播和后向传播
def profile_model(model, criterion, device, input_size=8):
    model.to(device)
    model.train()
    model_name = model.__class__.__name__
    
    # 创建随机输入数据
    inputs = torch.randn(32, input_size, device=device)
    targets = torch.randn(32, 1, device=device)
    
    # 使用profiler分析前向传播和后向传播
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("forward"):
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        
        with record_function("backward"):
            loss.backward()
    
    # 创建性能报告文件
    report_file = f"./profile_logs/{model_name}_profile_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs("./profile_logs", exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"\n{model_name}模型分析结果:\n")
        f.write("CPU & CUDA综合排序 (按CPU时间):\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        
        f.write("\nCPU & CUDA综合排序 (按CUDA时间):\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        f.write("\n各操作类型汇总:\n")
        f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    
    # 打印结果到终端
    print(f"\n{model_name}模型分析结果:")
    print("CPU & CUDA综合排序 (按CPU时间):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    print("\nCPU & CUDA综合排序 (按CUDA时间):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    print("\n各操作类型汇总:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    
    print(f"\n性能分析报告已保存至: {report_file}")
    
    return prof

class MathMLP(nn.Module):
    def __init__(self, input_size=8):
        super(MathMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128,72),
            nn.ReLU(),
            nn.Linear(72,1)
        )

    def forward(self, x):
        return self.layers(x)

class MathGroupKAN(nn.Module):
    def __init__(self, input_size=8):
        super(MathGroupKAN, self).__init__()
        self.layers = nn.Sequential(
            GroupKANLinear(input_size, 64,num_groups=4),
            nn.ReLU(),
            GroupKANLinear(64,32,num_groups=4),
            nn.ReLU(),
            GroupKANLinear(32,1,num_groups=4)
        )

    def forward(self, x):
        return self.layers(x)
    
class MathTaylorKAN(nn.Module):
    def __init__(self, input_size=8):
        super(MathTaylorKAN, self).__init__()
        self.layers = nn.Sequential(
            TaylorKANLinear(input_size, 65),
            nn.ReLU(),
            TaylorKANLinear(65,32),
            nn.ReLU(),
            TaylorKANLinear(32,1)
        )

    def forward(self, x):
        return self.layers(x)

# 定义复杂方程：结合多项式、三角函数、指数等非线性关系
def complex_equation(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    
    # 多项式项
    poly_term = 0.5 * x1**2 - 0.3 * x2**3 + 0.7 * x3 * x4**2
    
    # 三角函数项
    trig_term = 1.2 * np.sin(x5) * np.cos(x6)
    
    # 指数项和对数项
    exp_term = 0.4 * np.exp(0.1 * x7) - 0.5 * np.log(np.abs(x8) + 1)
    
    # 交互项
    interaction = 0.8 * x1 * x3 - 0.6 * x2 * x4 * x7 + 0.3 * x5 * x8
    
    # 分数项
    fraction_term = 1.5 / (1 + np.exp(-x1 - x2))
    
    # 组合所有项
    y = poly_term + trig_term + exp_term + interaction + fraction_term
    
    return y

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成随机输入数据 (10000个样本)
    n_samples = 10000
    X = np.random.randn(n_samples, 8) * 2  # 8个特征/变量，扩大取值范围

    # 计算每个样本的目标值
    y = np.array([complex_equation(x) for x in X])

    # 转换为DataFrame便于查看
    data = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(8)])
    data['y'] = y

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建并分析KAN模型
    groupkan_model = MathGroupKAN().to(device)
    # 创建并分析MLP模型
    mlp_model = MathMLP().to(device)

    # 分别对两个模型进行性能分析
    print("======= 单次前向和后向传播分析 =======")
    kan_prof = profile_model(groupkan_model, criterion, device)
    mlp_prof = profile_model(mlp_model, criterion, device)
    
    # 打印模型参数信息
    for model in [groupkan_model, mlp_model]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f'\n模型名称: {model.__class__.__name__}')
        print(f'模型总参数数量: {total_params:,}')
        print(f'可训练参数数量: {trainable_params:,}')
    
    # 是否进行完整训练
    run_full_training = True
    if run_full_training:
        # 使用性能分析训练KAN模型
        print("\n======= KAN模型训练 (启用Profiler) =======")
        kan_optimizer = optim.Adam(groupkan_model.parameters(), 1e-3)
        train(groupkan_model, train_loader, test_loader, criterion, kan_optimizer, epochs=1, profile_enabled=True)
        
        # 使用性能分析训练MLP模型
        # print("\n======= MLP模型训练 (启用Profiler) =======")
        # mlp_optimizer = optim.Adam(mlp_model.parameters(), 1e-3)
        # train(mlp_model, train_loader, test_loader, criterion, mlp_optimizer, epochs=1, profile_enabled=True)
        
        # 正常训练(可选)
        print("\n======= 正常训练过程 =======")
        print("KAN模型训练")
        kan_optimizer = optim.Adam(groupkan_model.parameters(), 1e-3)
        train(groupkan_model, train_loader, test_loader, criterion, kan_optimizer, epochs=20, profile_enabled=False)

