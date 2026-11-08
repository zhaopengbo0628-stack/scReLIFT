# scReLIFT 论文配套代码

本目录包含可安装的模型与运行入口、数据格式约定、环境配置，以及逐文件 SHA256 可追溯的论文源码快照。原始研究目录、冻结图版本、数据、结果和 checkpoint 均未修改。

先阅读 [README.md](README.md)、[数据格式](docs/DATA_FORMAT.md) 和 [论文流程索引](docs/PAPER_PIPELINES.md)。`src/screlift` 用于可配置路径的新工作流；`paper_snapshot` 保留论文原始脚本、参数与旧 namespace，其中历史脚本仍有服务器路径依赖。

流程为 SOURCE 数据准备 → 稳定调控模板 B → 细胞特异调控场 F → Teacher → RNA-only Student → 冻结 TARGET embedding → 读取标签评分 → 三 seed 汇总。新实验固定 seeds 20020130、20020131、20020132，报告均值和样本标准差（ddof=1）。历史五 seed/其他 seed 配置保留原文，不混入新的正式汇总。

标准训练环境为 Linux/CUDA。CPU 可用于推理及模型前向/反向核验。`--smoke` 运行两轮，输出标记为 smoke，不纳入论文汇总。

AD 四折、P2 donor/sample-group probe、PBMC 亚型与 GO/Reactome 分析的原始入口均保留并建立索引。这些专用流程需要原始数据、冻结划分及参考资源。本次整理未重跑论文正式实验；具体范围见 [复现说明](docs/REPRODUCIBILITY.md)。

许可证按用户选择设为 MIT。公开仓库为 [zhaopengbo0628-stack/scReLIFT](https://github.com/zhaopengbo0628-stack/scReLIFT)。论文 DOI 和数据/checkpoint 发布位置仍待补齐。
