import torch

# ==========================================
# 1. 为什么需要非线性？(The Problem)
# ==========================================
# 设定目标：我们希望 x=1.0 时，输出 0.73 (概率)
# 真实的逻辑是 Sigmoid 函数： y = 1 / (1 + e^(-wx))
x = torch.tensor([1.0])
y_true = torch.tensor([0.731]) # 这是 sigmoid(1.0) 的近似值

# 初始化权重 w
w = torch.tensor([0.1], requires_grad=True) # 随机给个初始值

print(f"目标: 输入 {x.item()} -> 输出应接近 {y_true.item()}")
print("-" * 30)

# ==========================================
# 训练循环 (引入 Activation)
# ==========================================
learning_rate = 0.5 # 稍微调大一点学习率
epochs = 100

for epoch in range(epochs):
    # --- 核心变化在这里 ---
    # 线性部分
    linear_output = w * x 
    
    # 非线性激活部分 (Activation Function)
    # Sigmoid 公式: 1 / (1 + exp(-x))
    # 它将 (-无穷, +无穷) 压缩到 (0, 1) 之间
    y_pred = torch.sigmoid(linear_output)
    
    # 计算损失 (MSE)
    loss = (y_pred - y_true) ** 2
    
    # 反向传播
    loss.backward()
    
    # 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.grad.zero_()
        
    if epoch % 1 == 0:
        print(f"Epoch {epoch}: w={w.item():.4f}, pred={y_pred.item():.4f}, Loss={loss.item():.6f}")

# ==========================================
# 验证：为什么要这么做？
# ==========================================
print("-" * 30)
print(f"训练结束，最终权重 w: {w.item():.4f}")

# 测试极端情况：输入 x = 10.0
x_test = torch.tensor([10.0])

# 情况 A: 如果我们只有线性模型 (y = w * x)
linear_result = w * x_test
# 情况 B: 我们现在的非线性模型 (y = sigmoid(w * x))
activation_result = torch.sigmoid(linear_result)

print(f"\n【验证非线性的必要性】当输入 x=10.0 时：")
print(f"1. 纯线性预测 (w*x): {linear_result.item():.2f} (错误！概率不可能大于1)")
print(f"2. 激活后预测 (Sigmoid): {activation_result.item():.4f} (正确！被限制在1.0以内)")