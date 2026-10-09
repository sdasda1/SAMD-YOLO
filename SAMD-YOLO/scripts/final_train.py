import sys
import os
import torch
import gc

# ======================================================
# 强制使用本地魔改版 ultralytics
# ======================================================

sys.path.insert(0, os.path.abspath("../sam_net"))

from ultralytics import YOLO

# ======================================================
# 主函数
# ======================================================

def main():

    print("🔥 SAMD-YOLO 多分辨率 + 多Seed 论文级训练启动")

    # ======================================================
    # 1. 推荐实验分辨率
    # ======================================================

    image_sizes = [
        512,
        640,
        1152
    ]

    # ======================================================
    # 2. 推荐 Seeds
    # ======================================================

    seeds_to_test = [
        42,
        3407,
        2024
    ]

    # ======================================================
    # 3. 自动循环
    # ======================================================

    total_exp = len(image_sizes) * len(seeds_to_test)
    exp_id = 0

    for imgsz in image_sizes:

        # 自动适配 batch
        if imgsz <= 512:
            batch_size = 4

        elif imgsz <= 768:
            batch_size = 4

        else:
            batch_size = 4

        for seed in seeds_to_test:

            exp_id += 1

            print("\n" + "=" * 70)
            print(f"🚀 Experiment {exp_id}/{total_exp}")
            print(f"📏 Resolution : {imgsz}")
            print(f"🎲 Seed       : {seed}")
            print(f"📦 Batch Size : {batch_size}")
            print("=" * 70 + "\n")

            # ======================================================
            # 重新加载模型
            # ======================================================

            model = YOLO(
                "../sam_net/sam_yolov8s.yaml",
                task='detect'
            ).load(
                "../weights/pretrained/yolov8s.pt"
            )

            # ======================================================
            # 开始训练
            # ======================================================

            results = model.train(

                # =========================
                # 数据集
                # =========================
                data="../detect_yuantu.yaml",

                # =========================
                # 基础训练
                # =========================
                epochs=200,
                imgsz=imgsz,
                batch=batch_size,
                workers=4,
                device=0,

                # =========================
                # 保存
                # =========================
                project="runs/MTL_MultiScale",

                name=f"SAMNet_{imgsz}_seed_{seed}",

                save=True,
                save_period=-1,
                plots=True,

                # =========================
                # 优化器
                # =========================
                optimizer="AdamW",

                lr0=8e-4,
                lrf=0.01,

                cos_lr=True,



                amp=True,

                # =========================
                # 数据增强
                # =========================

                hsv_h=0.02,
                hsv_s=0.35,
                hsv_v=0.35,

                translate=0.05,

                scale=0.35,

                fliplr=0.5,

                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
                degrees=0.0,

                # =========================
                # 重要！
                # =========================

                close_mosaic=0,

                cache=False,

                deterministic=True,

                seed=seed
            )

            print(f"\n✅ Resolution {imgsz} | Seed {seed} 训练完成")

            # ======================================================
            # 清理显存
            # ======================================================

            del model
            del results

            gc.collect()

            torch.cuda.empty_cache()

    print("\n🎉 全部分辨率实验完成！")


if __name__ == "__main__":
    main()