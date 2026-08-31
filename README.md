# r5 训练数据集 (dataset_full_depth_v2)

**内容**: 全量 2000 张 4ch PNG (BGR + log深度) + 校准标签 (赛方 new_labels_2000, 2026-08-31)

**构建** (Windows 本地):
```
python build_full_depth_dataset.py
```

**关键处理**:
- 深度解码 (专家评审修正): 16-bit PNG 直读毫米值; 8-bit JPG 用 v * 19999/255 线性映射 (旧版 *256 错误导致 21.7% 饱和)
- log 归一化: log(1+mm)/log(20000)*255, 0=无效深度
- 标签: 赛方校准版 new_labels_2000 (382/2000 文件改动, 总框 14417→15195, class_6/8 各 +378/+380)

**目录**: images/train/ (2000 PNG) + labels/train/ (2000 TXT)
