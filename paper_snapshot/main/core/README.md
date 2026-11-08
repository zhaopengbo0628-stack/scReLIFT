# 冻结共用核心

`run_b2_strict_lodo.py` 是已有冻结文件，SHA256：

`3de1ad3642e7af276658d9b239ccaf9457adf7f8e6078b95a237cf4605f2c7e1`

该文件包含 Teacher/Student、损失、训练/checkpoint 与评价实现。其余 `model/`、`loss/`、`training/`、`evaluation/` 是未来明确授权拆分时可用的目录，本次不复制或重写这些实现。

实际调控模块入口：

- `regulatory/s_mode/single_source.py`：现有单 Source B_S/F；保留原 `v5d` imports。
- `regulatory/common/regulatory_graph.py`：已有 TSS、peak、候选图操作。
- `regulatory/common/dmode_teacher.py`：已有 F 拟合操作。
- `regulatory/d_mode/dmode_reliable_selector.py`：Legacy 跨 Source B_D；没有用 CCRS 替换。

这些是 canonical source 的链接入口。现有 namespace/import 和 Linux runner 保持原样；文件分类本身不等于建立了新的可独立安装 Python 包。

`frozen_task35/` 指向 Task35 的原冻结源码快照；该历史快照继续只读。
