import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# SwanLab
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

# ==================== 配置区 ====================
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""

if IS_KAGGLE:
    TEST_DIR = "/kaggle/working/dataset/test"
    CHECKPOINT = "/kaggle/working/checkpoints/best_model.pth"
    NUM_WORKERS = 2
else:
    TEST_DIR = "dataset/test"
    CHECKPOINT = "checkpoints/best_model.pth"
    NUM_WORKERS = 4

BATCH_SIZE = 64
IMG_SIZE = 160
LABEL_SMOOTHING = 0.1
MODEL_SCALE = 1.5
# ===============================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 数据预处理 ====================
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==================== 模型架构（与 train.py 一致） ====================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

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
        return x + residual


class YogaCNN(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super().__init__()
        c = lambda x: int(x * scale)

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
        return self.fc2(x)


# ==================== 评估函数 ====================
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Testing")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return running_loss / total, correct / total, all_preds, all_labels


def main():
    print(f"设备: {DEVICE}")

    # 数据集
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    print(f"测试样本: {len(test_dataset)}")
    print(f"类别数: {len(test_dataset.classes)}")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)

    # 模型
    model = YogaCNN(num_classes=len(test_dataset.classes), scale=MODEL_SCALE).to(DEVICE)

    # 加载权重
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"加载模型: epoch={ckpt['epoch']}, best_val_acc={ckpt['best_acc']:.4f}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)

    print(f"\n========== 测试结果 ==========")
    print(f"测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Acc:  {test_acc:.4f} ({correct}/{total})")

    # SwanLab
    if SWANLAB_AVAILABLE:
        swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY", ""))
        swanlab.init(
            project="yoga-pose-classifier",
            experiment_name=f"scale{int(MODEL_SCALE*100)}_test",
            config={"model_scale": MODEL_SCALE, "checkpoint": CHECKPOINT},
        )
        swanlab.log({"test/loss": test_loss, "test/acc": test_acc})
        swanlab.finish()


if __name__ == "__main__":
    main()
