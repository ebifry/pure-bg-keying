#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
源引导补缺失 · Restore by Source
================================================================================
一句话：**被误扣掉的角色像素，回源头去查它到底是不是角色。**

⭐ 为什么需要它
    混合方案（A ∩ B）的原则是"任何一方说是背景就扣"。这条原则的代价是：
    对**细长 / 半透明**结构（袜子边缘、鞋带、发丝）它会把角色也扣掉。

    实例：走路序列某一帧的脚踝处出现一个三角缺口。
          ⛔ 靠"看起来不对劲"是查不出来的 —— 得回源视频量一下：
          那一块"背景占比只有 7.4%，均色是袜子色" → **那是角色，是被误扣的。**

⭐ 治法（⛔ 不是猜、不是多数表决，是查源头）
    ① 源帧和抠好的帧一一对应（同一条链下来的，顺序就是对应关系）
    ② 用两者的**角色外框**做比例映射，把抠好帧的坐标换算回源帧坐标
    ③ 源帧那里"不是背景色"、而抠好帧那里"透明/半透明" → ⭐ 判定为被误扣 → 恢复
    ④ 恢复出来的颜色取该帧**最近的实心像素**色（⛔ 不取原像素色，否则又把背景混色引回来）

⭐⭐ 只补不删 —— 绝不动已经实心的像素。所以它**不可能把结果改坏**，
   最坏情况是"没补到东西"，不会新增漏抠。

⚠️ 前提：两条序列得**一一对应**。如果源帧和成品帧不是一一对应（比如中间经过抽帧、
   缩放、重排），得先自己做帧匹配，本工具不管这件事。

⭐ 用法
    python restore_by_source.py <源帧目录> <抠好的目录> <输出目录>
    python restore_by_source.py src/ keyed/ fixed/ --tol 42 --near 6

⭐ 内置自检：逐帧报恢复了多少像素 + 总量占角色面积的比例
   ⛔ 比例异常高（比如 >5%）说明不是"补缺"，是**映射错了** —— 别直接用结果。

License: MIT
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy import ndimage

BG_TOL = 42         # 距背景多近算"背景"（源帧一侧的判据）
NEAR_PX = 6         # ⭐ 只补「贴着已有实心区」的像素，多远以内算贴着
WARN_RATIO = 0.05   # ⭐ 单帧恢复量占角色面积超过这个比例 → 报警（多半是映射错了）


def bg_of(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    c = np.concatenate([rgb[:40, :40].reshape(-1, 3), rgb[:40, -40:].reshape(-1, 3),
                        rgb[-40:, :40].reshape(-1, 3), rgb[-40:, -40:].reshape(-1, 3)])
    return np.median(c, axis=0)


def is_bg(rgb: np.ndarray, bg: np.ndarray, tol: int = BG_TOL) -> np.ndarray:
    return np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).sum(axis=2) <= tol


def restore_frame(src_rgb: np.ndarray, keyed_rgba: np.ndarray, bg: np.ndarray | None = None,
                  tol: int = BG_TOL, near_px: int = NEAR_PX):
    """恢复单帧被误扣的角色像素。返回 (新的 RGBA, 报告 dict)"""
    bg = bg_of(src_rgb) if bg is None else np.asarray(bg)
    A = keyed_rgba
    solid = A[:, :, 3] > 128
    if not solid.any():
        return A, dict(restored=0, area=0, ratio=0.0, note="整帧全透明，跳过")

    # ---- ① 两条序列的角色外框 ----
    ky, kx = np.where(solid)
    src_mask = ~is_bg(src_rgb, bg, tol)
    if not src_mask.any():
        return A, dict(restored=0, area=int(solid.sum()), ratio=0.0, note="源帧全是背景色")
    sy, sx = np.where(src_mask)

    # ---- ② 抠好帧坐标 → 源帧坐标（按外框等比映射）----
    Y, X = np.mgrid[0:A.shape[0], 0:A.shape[1]]
    gy = sy.min() + (Y - ky.min()) * (sy.max() - sy.min() + 1) / (ky.max() - ky.min() + 1)
    gx = sx.min() + (X - kx.min()) * (sx.max() - sx.min() + 1) / (kx.max() - kx.min() + 1)
    gy = np.clip(np.round(gy).astype(int), 0, src_rgb.shape[0] - 1)
    gx = np.clip(np.round(gx).astype(int), 0, src_rgb.shape[1] - 1)

    # ---- ③ 源帧是角色 且 这边不是实心 → 要恢复 ----
    want = src_mask[gy, gx]
    # ⭐ 只补贴着已有实心区的（否则会把远处的背景碎块也拉进来）
    near = ndimage.binary_dilation(solid, iterations=max(1, near_px))
    restore = want & (~solid) & near
    area = int(solid.sum())
    if not restore.any():
        return A, dict(restored=0, area=area, ratio=0.0)

    # ---- ④ 颜色取最近的实心像素（⛔ 避免把背景混色又引回来）----
    fixed = solid | restore
    _, inds = ndimage.distance_transform_edt(~fixed, return_indices=True)
    near_rgb = A[:, :, :3][inds[0], inds[1]]
    out = A.copy()
    out[:, :, :3] = np.where(restore[:, :, None], near_rgb, A[:, :, :3])
    out[:, :, 3] = np.where(restore, 255, A[:, :, 3])
    n = int(restore.sum())
    return out, dict(restored=n, area=area, ratio=n / max(area, 1))


def main():
    ap = argparse.ArgumentParser(description="源引导补缺失：只补不删")
    ap.add_argument("src_dir", help="源帧目录（原始画面）")
    ap.add_argument("keyed_dir", help="抠好的目录（RGBA）")
    ap.add_argument("out_dir", help="输出目录")
    ap.add_argument("--pattern", default="*.png")
    ap.add_argument("--tol", type=int, default=BG_TOL, help=f"源帧判背景的容差（默认 {BG_TOL}）")
    ap.add_argument("--near", type=int, default=NEAR_PX, help=f"贴着实心区多少 px 内才补（默认 {NEAR_PX}）")
    a = ap.parse_args()

    srcs = sorted(glob.glob(os.path.join(a.src_dir, a.pattern)))
    keys = sorted(glob.glob(os.path.join(a.keyed_dir, a.pattern)))
    if not srcs or not keys:
        print("⛔ 两边的目录里都得有匹配到的帧")
        sys.exit(1)
    if len(srcs) != len(keys):
        print(f"⛔ 帧数不一致（源 {len(srcs)} / 抠好 {len(keys)}）—— ⚠️ 两条序列必须一一对应")
        sys.exit(1)
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"[源] {len(srcs)} 帧   [抠好] {len(keys)} 帧   →  {a.out_dir}")
    t0, tot, warn, area0 = time.time(), 0, 0, 0
    for i, (fs, fk) in enumerate(zip(srcs, keys)):
        src = np.array(Image.open(fs).convert("RGB"))
        keyed = np.array(Image.open(fk).convert("RGBA"))
        out, rep = restore_frame(src, keyed, tol=a.tol, near_px=a.near)
        Image.fromarray(out).save(os.path.join(a.out_dir, os.path.basename(fk)))
        tot += rep["restored"]
        area0 = max(area0, rep["area"])
        if rep["ratio"] > WARN_RATIO:
            warn += 1
            print(f"  [{i}] ⚠️ 恢复 {rep['restored']}px = 角色面积 {rep['ratio']:.1%}"
                  f"  ← 偏高，检查两条序列是不是真的对应")
        elif rep["restored"]:
            print(f"  [{i}] 恢复 {rep['restored']:>6}px（{rep['ratio']:.3%}）")
    print(f"\n[完成] 合计恢复 {tot}px   用时 {time.time()-t0:.1f}s")
    if warn:
        print(f"⛔ 有 {warn} 帧超出 {WARN_RATIO:.0%} 警戒线 —— ⚠️ 先确认对应关系，再决定用不用这些结果")
    else:
        print("⭐ 只补不删：实心区一个像素都没被改过")


if __name__ == "__main__":
    main()
