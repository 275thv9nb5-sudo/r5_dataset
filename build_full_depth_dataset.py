"""
Build FULL-data RGB+Depth 4ch dataset — v2 (2026-08-31)
========================================================
Sources:
  - images: 修正数据集 (visible + depth, 16-bit corrected decode)
  - labels: 赛方校准后的 new_labels_2000 (382 files changed, +778 boxes;
    class_6 +378 / class_8 +380 vs old labels)
Depth decode (expert-review fix, lesson #40):
  - 16-bit PNG depth: raw mm values directly
  - 8-bit JPG depth: v * (19999/255) linear-to-mm
Then log-normalize: log(1+mm)/log(20000)*255, uint8, valid mask depth>0.

Output: output/dataset_full_depth_v2/{images,labels}/train = all 2000 images.

Usage (LOCAL, Windows):
  python scripts/build_full_depth_dataset.py
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

PROJECT = Path(r"E:\aic_project")
RAW = PROJECT / "初赛数据集-面向城市场景的多模态目标检测修正/训练集/AIC2026_Train_2000"
VIS_DIR = RAW / "visible"
DEPTH_DIR = RAW / "depth"
LBL_SRC = RAW.parent / "new_labels_2000"          # 校准标签在 训练集/new_labels_2000 (不在AIC2026_Train_2000内!)
DST = PROJECT / "output/dataset_full_depth_v2"

DEPTH_MAX_MM = 20000.0


def imread_cn(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def depth_to_log8(depth_raw):
    """raw depth (uint16 PNG or uint8 JPG) -> log-normalized uint8 channel."""
    if depth_raw.dtype == np.uint16:
        depth_mm = depth_raw.astype(np.float32)          # already mm
    else:
        # 8-bit JPG: linear to mm (v * 19999/255) — fixed 2026-08-29
        depth_mm = depth_raw.astype(np.float32) * (19999.0 / 255.0)
    valid = depth_mm > 0
    log_depth = np.zeros_like(depth_mm, dtype=np.float32)
    log_depth[valid] = np.log1p(depth_mm[valid]) / np.log(DEPTH_MAX_MM + 1) * 255.0
    return np.clip(log_depth, 0, 255).astype(np.uint8)


def build_one(stem, ext_map):
    vis = imread_cn(str(VIS_DIR / f"{stem}{ext_map['visible'].get(stem, '.jpg')}"))
    dep = imread_cn(str(DEPTH_DIR / f"{stem}{ext_map['depth'].get(stem, '.jpg')}"),
                    cv2.IMREAD_UNCHANGED)
    if vis is None or dep is None:
        return None
    if dep.ndim == 3:
        dep = dep[:, :, 0]              # 8-bit JPG: 3 identical channels
    if dep.shape[:2] != vis.shape[:2]:
        dep = cv2.resize(dep, (vis.shape[1], vis.shape[0]))
    dep8 = depth_to_log8(dep)
    return np.concatenate([vis, dep8[..., None]], axis=-1)  # (h, w, 4) uint8


def main():
    # extension maps
    ext_map = {}
    for mod, d in (("visible", VIS_DIR), ("depth", DEPTH_DIR)):
        ext_map[mod] = {}
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            ext_map[mod][stem] = ext

    (DST / "images/train").mkdir(parents=True, exist_ok=True)
    (DST / "labels/train").mkdir(parents=True, exist_ok=True)

    n = 0
    for lbl in sorted(LBL_SRC.glob("*.txt")):
        stem = lbl.stem
        # link label (new_labels_2000 = 赛方校准标签)
        dst_lbl = DST / "labels/train" / lbl.name
        try:
            os.link(str(lbl), str(dst_lbl))
        except OSError:
            shutil.copy2(str(lbl), str(dst_lbl))
        # build image
        img4 = build_one(stem, ext_map)
        if img4 is None:
            print(f"[WARN] cannot build {stem}, skipping")
            continue
        cv2.imwrite(str(DST / "images/train" / f"{stem}.png"), img4)
        n += 1
        if n % 200 == 0:
            print(f"  {n}/2000")

    assert n == 2000, f"expected 2000, got {n}"
    print(f"[OK] dataset_full_depth built: {n} images + labels")

    # sanity: check a small (JPG) image depth channel is no longer saturated
    test = cv2.imdecode(np.fromfile(str(DST / "images/train" / "00000004.png"),
                                    dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    dep = test[:, :, 3]
    print(f"[Sanity] 00000004 depth channel: saturated(>=254)={np.mean(dep >= 254) * 100:.1f}% "
          f"(was 21.7% with old decode)")
    assert np.mean(dep >= 254) < 0.02, "depth channel still saturated!"


if __name__ == '__main__':
    main()
