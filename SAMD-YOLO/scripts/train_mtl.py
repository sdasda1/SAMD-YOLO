import sys
import os
import torch
import gc
import matplotlib
matplotlib.use('Agg') # 或者 'TkAgg', 优先尝试 'Agg'


# 强行指定路径，确保使用的是我们本地魔改版的 sam_net/ultralytics！
sys.path.insert(0, os.path.abspath("../sam_net"))
from ultralytics import YOLO


def main():
    print("🔥 启动 [M5 终局之战：软标签 + 频域空间余弦蒸馏 (F-CSD)] 200轮终极训练 (多Seed版本)...")

    # 1. 定义你要测试的 5 个 Seed 
    # (包含了深度学习界经典的 42, 以及近年来被证明容易出好结果的 3407)
    seeds_to_test = [1, 2, 3, 4, 5]

    for i, seed in enumerate(seeds_to_test):
        print(f"\n" + "=" * 60)
        print(f"🚀 开始训练第 {i + 1}/5 个模型 | 当前 Seed: {seed}")
        print(f"=" * 60 + "\n")

        # 2. ⚠️ 核心：必须在循环内部重新加载模型，确保每次都从头开始！
        model = YOLO("../sam_net/sam_yolov8s.yaml", task='detect').load("../weights/pretrained/yolov8s.pt")

        # 3. 核心超参数配置
        results = model.train(
            data=r"C:\Users\Zhl\Desktop\ypl\MTL\detect_yuantu.yaml",  # 你的数据集配置文件
            epochs=200,  # 满血 200 轮
            imgsz=1152,  # 高分辨率输入
            workers=4,  # A5000 搭配 4 个 worker 最佳

            # ⚠️ 【显存警告】如果训练时报 CUDA Out of Memory，请毫不犹豫地将 batch 改为 2！
            batch=4,
            device=0,
            plots=True,
            # ⚠️ 核心：动态保存路径，避免覆盖
            name=f"SAM_Net_Final_M5_6_seed_{seed}",
            project="runs/MTL_Final",

            # ===== 优化器策略 =====
            optimizer="AdamW",
            lr0=8e-4,  # 学习率起点
            lrf=0.01,  # 余弦退火的最低点 (lr0 * 0.01)
            cos_lr=True,  # 开启余弦退火
            amp=True,  # 开启混合精度加速


            # ===== 颜色增强 (绝对安全，拉满) =====
            hsv_h=0.02,  # 色调变化
            hsv_s=0.35,  # 饱和度变化
            hsv_v=0.35,  # 亮度变化 (对工业件极其重要)

            # ===== 空间增强 (为了软标签对齐，必须全部设为 0) =====
            translate=0.05,
            scale=0.35,
            fliplr=0.5,
            flipud=0.0,  # 严禁上下翻转
            mosaic=0.0,  # 严禁 Mosaic
            mixup=0.0,  # 严禁 Mixup
            degrees=0.0,  # 严禁旋转
            copy_paste=0.0,  # 严禁复制粘贴

            # ===== 其他配置 =====
            cache=False,  # A5000 读图够快，设为 False 稳妥
            save=True,  # 保存最佳模型
            save_period=-1,  # 不保存中间 epoch，省硬盘
            deterministic=True,  # 开启确定性算法
            seed=seed  # 🌟 动态传入当前的 seed！
        )

        print(f"\n✅ Seed {seed} 训练结束！")

        # 4. ⚠️ 核心：彻底清空显存与内存，为下一个 Seed 腾出 24G 空间
        del model
        del results
        gc.collect()
        torch.cuda.empty_cache()

    print("\n🎉🎉🎉 全部 5 个 Seed 的满血训练已顺利完成！论文核心数据已拿到！")


if __name__ == "__main__":
    main()