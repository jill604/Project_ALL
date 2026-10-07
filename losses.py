import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    """
    专为 TinyCD 设计的无坑版混合损失函数:
    Total_Loss = alpha * BCE + (1 - alpha) * Dice
    """
    def __init__(self, alpha=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha          # BCE 权重 (默认 0.5)
        self.smooth = smooth        # 平滑因子，防止分母除以 0
        self.bce = nn.BCELoss()     # 注意：因为输出自带 Sigmoid，必须用 BCELoss！

    def forward(self, preds, targets):
        """
        :param preds: 模型预测概率图 [B, 1, H, W]，值域 [0, 1]
        :param targets: 真实标签图 [B, 1, H, W] 或 [B, H, W]，值域 {0, 1}
        """
        # ==========================================
        # 1. 维度与类型对齐 (防报错)
        # ==========================================
        # train.py 里模型输出已被 squeeze 成 [B, H, W], preds 与 targets 都需
        # 对齐到 [B, 1, H, W], 否则 BCELoss 因形状不一致直接报错
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        # 数值截断保护：防止出现极值 0.0 或 1.0 导致 BCE 计算出 NaN
        preds = torch.clamp(preds, min=1e-7, max=1.0 - 1e-7)

        # ==========================================
        # 2. 计算标准 BCE 损失
        # ==========================================
        bce_loss = self.bce(preds, targets)

        # ==========================================
        # 3. 计算 Dice 损失 (强行惩罚漏检，拉高 Recall)
        # ==========================================
        # 将矩阵展平为一维向量计算交集与并集
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # ==========================================
        # 4. 加权融合
        # ==========================================
        total_loss = self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss
        return total_loss


# =========================================================================
# 独立单元测试 —— 运行此文件，确保 Loss 绝对正常
# =========================================================================
if __name__ == "__main__":
    print("🚀 正在测试 BCEDiceLoss 是否正常运转...")
    criterion = BCEDiceLoss(alpha=0.5)

    # 模拟网络输出的概率图 [2, 1, 256, 256]，值域 [0, 1]
    fake_preds = torch.rand(2, 1, 256, 256, requires_grad=True)
    # 模拟真实标签
    fake_targets = torch.randint(0, 2, (2, 1, 256, 256)).float()

    loss = criterion(fake_preds, fake_targets)
    print(f"计算出的 Loss 值为: {loss.item():.4f}")

    # 测试反向传播是否顺畅
    loss.backward()
    assert fake_preds.grad is not None, "❌ 错误：Loss 无法回传梯度！"
    print("🎉 单元测试通过！Loss 数值正常，反向传播顺畅！")