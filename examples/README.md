# 示例素材

## 文件

| 文件 | 说明 |
|---|---|
| `demo_input.png` | 待抠的素材（纯色背景） |
| `demo_output.png` | 本工具抠出的透明底 PNG（RGBA） |
| `compare.png` | 对照图：原图 / 透明底（青底显，验漏抠）/ 透明底（中灰底显，验观感） |

## 关于 demo 素材的来源

Demo 使用的是一张**已废弃的早期角色草稿**（不再用于生产），仅作技术演示。

⚠️ **它仍属于原作方的角色资产**，因此：

- ✅ 可用于演示本工具的效果
- ⛔ **不得**作为素材二次分发或商用
- ⭐ **如果你 fork 本仓库并用于发布**，建议把你自己的 demo 素材换掉

## 换掉示例的方法

```bash
# 用你自己的图覆盖 demo_input.png，然后重跑
python pure_bg_keying.py examples/demo_input.png examples/demo_output.png
```

## 复现对照图

```python
from PIL import Image

src = Image.open("examples/demo_input.png").convert("RGB")
res = Image.open("examples/demo_output.png").convert("RGBA")


def on(color):
    bg = Image.new("RGBA", res.size, color + (255,))
    bg.alpha_composite(res)
    return bg.convert("RGB")


on((0, 220, 255))     # ⭐ 青底 —— 验「漏抠/残留背景」
on((128, 128, 128))   # ⭐ 中灰底 —— 验「边缘观感」（桌面真实环境）
```

> ⭐ 对照图里**两种底色都要放** —— 原因见 [../docs/VALIDATION.md](../docs/VALIDATION.md#一底色要选能和错误对比出来的)。
