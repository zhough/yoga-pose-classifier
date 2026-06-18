import os
import shutil
import random

# ============ 配置区 ============
IS_KAGGLE = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""

if IS_KAGGLE:
    DATASET_DIR = "/kaggle/input/datasets/shrutisaxena/yoga-pose-image-classification-dataset"
    TRAIN_DIR = "/kaggle/working/dataset/train"
    VAL_DIR = "/kaggle/working/dataset/val"
    TEST_DIR = "/kaggle/working/dataset/test"
else:
    DATASET_DIR = "dataset"
    TRAIN_DIR = "dataset/train"
    VAL_DIR = "dataset/val"
    TEST_DIR = "dataset/test"

TRAIN_RATIO = 0.7                  # 训练集占比
VAL_RATIO = 0.15                   # 验证集占比
TEST_RATIO = 0.15                  # 测试集占比
RANDOM_SEED = 42                   # 随机种子，保证结果可复现
MOVE_MODE = False                  # True=移动文件, False=复制文件
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
# ===============================

random.seed(RANDOM_SEED)


def find_dataset_root(root_dir):
    """自动探测实际的类别文件夹层级（处理嵌套目录）"""
    exclude = {"train", "val", "test", "__MACOSX"}
    subdirs = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and d not in exclude]

    if len(subdirs) == 0:
        return root_dir  # 没有子目录，原样返回

    # 检查当前层级是否直接包含图片（即这就是类别文件夹层级）
    for d in subdirs[:3]:  # 抽样检查前3个
        for f in os.listdir(os.path.join(root_dir, d)):
            if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS:
                return root_dir  # 已找到类别文件夹

    # 当前层级没有图片，可能是外层包裹目录，尝试深入一层
    if len(subdirs) == 1:
        inner = os.path.join(root_dir, subdirs[0])
        print(f"自动探测: 进入子目录 '{subdirs[0]}'")
        return find_dataset_root(inner)

    return root_dir


def get_class_folders(root_dir):
    """获取所有类别文件夹名（排除 train/val 等非类别目录）"""
    exclude = {"train", "val", "test"}
    return sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in exclude
    ])


def get_image_files(class_dir):
    """获取文件夹下所有图片文件"""
    files = []
    for f in os.listdir(class_dir):
        if os.path.isfile(os.path.join(class_dir, f)):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                files.append(f)
    return files


def split_and_copy():
    actual_dir = find_dataset_root(DATASET_DIR)
    print(f"数据集根目录: {actual_dir}")
    class_folders = get_class_folders(actual_dir)
    print(f"发现 {len(class_folders)} 个类别文件夹\n")

    total_train = 0
    total_val = 0
    total_test = 0

    for cls in class_folders:
        cls_src = os.path.join(actual_dir, cls)
        images = get_image_files(cls_src)

        if len(images) == 0:
            print(f"[跳过] {cls} — 没有图片文件")
            continue

        random.shuffle(images)

        n_train = max(1, int(len(images) * TRAIN_RATIO))
        n_val = max(1, int(len(images) * VAL_RATIO))
        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for split_dir, files in [
            (TRAIN_DIR, train_files),
            (VAL_DIR, val_files),
            (TEST_DIR, test_files),
        ]:
            dst_cls = os.path.join(split_dir, cls)
            os.makedirs(dst_cls, exist_ok=True)
            for f in files:
                src = os.path.join(cls_src, f)
                dst = os.path.join(dst_cls, f)
                if MOVE_MODE:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)

        print(f"{cls}: {len(train_files)} train | {len(val_files)} val | {len(test_files)} test")
        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

    print(f"\n========== 完成 ==========")
    print(f"训练集总计: {total_train} 张")
    print(f"验证集总计: {total_val} 张")
    print(f"测试集总计: {total_test} 张")
    print(f"训练集路径: {TRAIN_DIR}/")
    print(f"验证集路径: {VAL_DIR}/")
    print(f"测试集路径: {TEST_DIR}/")


if __name__ == "__main__":
    split_and_copy()
