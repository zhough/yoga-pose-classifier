import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# SwanLab 实验追踪（未安装时安全跳过）
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("⚠ SwanLab 未安装，跳过实验记录。pip install swanlab")

# ==================== 配置区 ====================
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""

if IS_KAGGLE:
    TRAIN_DIR = "/kaggle/working/dataset/train"
    VAL_DIR = "/kaggle/working/dataset/val"
    TEST_DIR = "/kaggle/working/dataset/test"
    SAVE_DIR = "/kaggle/working/checkpoints"
    NUM_WORKERS = 2
else:
    TRAIN_DIR = "dataset/train"
    VAL_DIR = "dataset/val"
    TEST_DIR = "dataset/test"
    SAVE_DIR = "checkpoints"
    NUM_WORKERS = 4

NUM_CLASSES = 98
BATCH_SIZE = 64
EPOCHS = 200                              # 更多迭代
LEARNING_RATE = 3e-2                      # SGD 从零训练 CNN 常用 0.01~0.1
IMG_SIZE = 160                            # 降分辨率，减少过拟合，加速训练
MODEL_NAME = "yoga_cnn"                   # "yoga_cnn" / "resnet50" / "resnet18"
MODEL_SCALE = 1.5               # 模型缩放因子: 0.5=小型, 1.0=标准, 2.0=大型
WEIGHT_DECAY = 1e-3
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0            # MixUp 强度，0=关闭


def get_device():
    """安全检测可用设备，GPU 不兼容时回退 CPU"""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        # 用一次 conv2d 验证 GPU 算力兼容性
        x = torch.randn(1, 3, 64, 64, device="cuda")
        w = torch.randn(8, 3, 3, 3, device="cuda")
        _ = torch.nn.functional.conv2d(x, w)
        del x, w
        torch.cuda.empty_cache()
        return torch.device("cuda")
    except Exception as e:
        print(f"⚠ GPU 不兼容: {e}")
        print("自动回退到 CPU 训练")
        torch.cuda.empty_cache()
        return torch.device("cpu")


DEVICE = get_device()
# ===============================================


# ==================== 数据预处理 ====================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ==================== CNN 模型 ====================
class ConvBlock(nn.Module):
    """带残差连接的卷积块"""

    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # skip: 1×1 conv 调通道（in_c≠out_c）+ AvgPool2d 下采样
        if in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.AvgPool2d(2, 2),
            )
        else:
            self.skip = nn.AvgPool2d(2, 2)

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x + residual
        return x


class YogaCNN(nn.Module):
    """带残差连接的 CNN 用于瑜伽体式分类"""

    def __init__(self, num_classes=NUM_CLASSES, scale=MODEL_SCALE):
        super().__init__()
        c = lambda x: int(x * scale)  # 通道缩放

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(64), kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c(64)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )

        self.block1 = ConvBlock(c(64), c(64), dropout=0.0)
        self.block2 = ConvBlock(c(64), c(128), dropout=0.1)
        self.block3 = ConvBlock(c(128), c(256), dropout=0.2)
        self.block4 = ConvBlock(c(256), c(512), dropout=0.2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(c(512), c(256))
        self.bn_fc = nn.BatchNorm1d(c(256))
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(c(256), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x


# ==================== 训练函数 ====================
def mixup_data(x, y, alpha):
    """MixUp 数据增强：将两张图按比例混合"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp 损失：两路损失的加权和"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if MIXUP_ALPHA > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

        optimizer.zero_grad()
        outputs = model(images)

        if MIXUP_ALPHA > 0:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            _, preds = torch.max(outputs, 1)
            correct += (lam * (preds == labels_a).float() +
                        (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{correct/total:.3f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validating", leave=False)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{correct/total:.3f}")

    return running_loss / total, correct / total


def main():
    print(f"设备: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据集
    print("加载数据集...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    print(f"类别数: {len(train_dataset.classes)}")
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 模型
    num_classes = len(train_dataset.classes)
    if MODEL_NAME == "yoga_cnn":
        model = YogaCNN(num_classes=num_classes, scale=MODEL_SCALE)
    else:
        from torchvision import models
        weights = getattr(models, {
            "resnet18": "ResNet18_Weights",
            "resnet50": "ResNet50_Weights",
        }[MODEL_NAME]).DEFAULT
        model = getattr(models, MODEL_NAME)(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4
    )

    # ============ SwanLab 初始化 ============
    if SWANLAB_AVAILABLE:
        api_key = ""

        # 尝试从 Kaggle Secrets 读取
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("SWANLAB_API_KEY")
        except Exception:
            pass

        # 回退：从环境变量读取（本地使用）
        if not api_key:
            api_key = os.environ.get("SWANLAB_API_KEY", "")

        if api_key:
            swanlab.login(api_key=api_key)

        swanlab.init(
            project="yoga-pose-classifier",
            experiment_name=f"scale{int(MODEL_SCALE*100)}",
            config={
                "architecture": MODEL_NAME,
                "model_scale": MODEL_SCALE,
                "mixup_alpha": MIXUP_ALPHA,
                "num_classes": len(train_dataset.classes),
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "img_size": IMG_SIZE,
                "optimizer": "SGD",
                "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY,
                "scheduler": "CosineAnnealingLR",
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "device": str(DEVICE),
            },
            logdir=os.path.join(SAVE_DIR, "swanlab"),
        )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("开始训练...\n")

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"===== Epoch {epoch}/{EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"当前学习率: {current_lr:.2e}")

        # SwanLab 记录
        if SWANLAB_AVAILABLE:
            swanlab.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "train/lr": current_lr,
            })

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "classes": train_dataset.classes,
            }, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f">>> 保存最佳模型 (acc={best_acc:.4f})")

        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "classes": train_dataset.classes,
            }, os.path.join(SAVE_DIR, f"checkpoint_epoch{epoch}.pth"))

        print()

    print(f"========== 训练完成 ==========")
    print(f"最佳验证准确率: {best_acc:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))

    # 保存类别映射
    with open(os.path.join(SAVE_DIR, "classes.txt"), "w", encoding="utf-8") as f:
        for cls in train_dataset.classes:
            f.write(cls + "\n")

    # 结束 SwanLab
    if SWANLAB_AVAILABLE:
        swanlab.finish()


if __name__ == "__main__":
    main()
