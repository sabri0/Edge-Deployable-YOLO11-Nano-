#!/usr/bin/env python3
"""Fine-tune yolo11n-pose on the UCO SLR slice (subject-disjoint train/val)."""
import torch
from ultralytics import YOLO

if __name__ == "__main__":
    dev = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11n-pose.pt")        # transfer from COCO-pretrained
    model.train(
        data="ft_dataset/uco_slr.yaml",
        epochs=45, imgsz=640, batch=8, device=dev,
        workers=2, patience=15, seed=0, close_mosaic=8,
        project="ft_runs", name="yolo11n_uco_slr", exist_ok=True,
        pose=18.0,            # emphasise keypoint loss
        plots=False, verbose=True,
    )
    print(">> best weights: ft_runs/yolo11n_uco_slr/weights/best.pt")
