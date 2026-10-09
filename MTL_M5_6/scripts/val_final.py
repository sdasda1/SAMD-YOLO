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
    print("🚀 正在加载你辛苦炼制了 200 轮的终极权重...")
    # ⚠️ 请检查下面这个路径，确保它指向你刚才训练保存的 best.pt
    # 通常在 runs/detect/dual_stream_final/weights/best.pt
    model = YOLO("runs/MTL_Final/SAM_Net_Final_M5_6_seed_2/weights/best.pt")

    print("📊 正在全速进行验证集评估，准备生成最终总结表...")
    # 执行验证
    # 执行验证 (参数完全对齐训练时的默认行为)
    model.val(
        data=r"C:\Users\Zhl\Desktop\ypl\detect_yuantu_test.yaml",
        imgsz=1152,  # ⚠️ 极其关键：必须改成你【训练时】用的图片尺寸！(比如 512, 640 或 1024)
        split="val",
        batch=4,
        half=True,  # 开启半精度，对齐训练时的 AMP
        conf=0.001,  # 极低置信度阈值，用于画出完整的 PR 曲线，刷出最高 mAP
        iou=0.6,# NMS 的 IoU 阈值，YOLOv8 默认是 0.6 或 0.7
        device=0,  # 明确指定显卡

    # 🌟 自定义保存路径的核心参数
        project = r"C:\Users\Zhl\Desktop\ypl\MTL_M5_6\scripts\runs\detect",  # 你想保存到的任意本地文件夹
        name = "M5_6_Seed2_Test_1152_1",
    )
    print("✅ 验证完成！所有的 PR 曲线、混淆矩阵等图表已经重新生成并保存在了 runs/detect/val 中！")

if __name__ == '__main__':
    main()