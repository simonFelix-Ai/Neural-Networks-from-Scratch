import torch

# ==========================================
# 1. 数据准备 (Data Preparation)
# ==========================================
# 目标：拟合 y = 3 * x
# 输入 x = 1.0, 目标 y = 3.0
x = torch.tensor([1.0]) 
y_true = torch.tensor([3.0])

# ==========================================
# 2. 权重初始化 (Weight Initialization)
# ==========================================
# 我们随机初始化一个权重 w
w = torch.tensor([1.0])

print(f"训练开始前 -> 输入: {x.item()}, 初始权重: {w.item():.2f}, 初始预测: {x.item() * w.item():.2f}")

# 超参数
learning_rate = 0.1
epochs = 50

print("-" * 30)

# ==========================================
# 训练循环
# ==========================================
for epoch in range(epochs):
    # 3. 前向传播 (Forward Pass)
    # 计算图构建：y_pred = w * x
    # 在这里，激活函数可以看作是 f(x) = x (线性激活)
    y_pred = w * x
    
    # 4. 计算损失 (Loss Calculation)
    # 使用均方误差 (MSE): Loss = (y_pred - y_true)^2
    loss = (y_pred - y_true) ** 2
    
    # 5. 参数更新 (Parameter Update)
    # 梯度下降公式: w_new = w_old - lr * gradient
    grad_w = 2 * (y_pred - y_true) * x
    w -= learning_rate * grad_w

    # 打印日志
    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, w = {w.item():.4f}, w.grad = {w.grad}")

print("-" * 30)
print(f"训练结束后 -> 最终权重 w: {w.item():.4f}")
print(f"最终预测: {x.item()} * {w.item():.4f} = {x.item() * w.item():.4f}")