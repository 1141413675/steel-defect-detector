"""在 NEU-DET 验证集上重跑检测指标。

用法：
    python eval.py --weights weights/best.pt --data data.yaml

输出总体 mAP50、mAP50-95，以及每个类别的 mAP50，和 README 中的统计口径一致。
"""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="评估钢材表面缺陷检测模型")
    parser.add_argument("--weights", default="weights/best.pt", help="权重路径")
    parser.add_argument("--data", default="data.yaml", help="数据集配置")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--split", default="val", help="评估划分：train / val / test")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split)

    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print("按类别：")
    for idx, class_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(class_id)]
        print(f"  {name}: mAP50 = {metrics.box.ap50[idx]:.3f}")


if __name__ == "__main__":
    main()
