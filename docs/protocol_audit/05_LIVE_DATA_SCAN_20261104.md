# AD、PBMC、BMMC 服务器实时数据扫描

扫描日期：2026-11-04。SSH：`zpb`，主机 `n1`，账号 `u202522802045`。服务器项目根目录：`/home/u202522802045/share/zpb`。下列相对路径均以此为根。此次只读取文件，没有修改服务器代码、数据、模型或结果。

读取范围：正式数据 registry、逐单位清单、预处理完成记录、RNA/ATAC 矩阵头、已保存特征数组及关键实现。没有重新遍历全部原始 counts，也没有重新校验全部大文件 SHA256；代码快照 SHA256 与读取的结构化数据保存在 [live_data_inventory_20261104.json](live_data_inventory_20261104.json)。

本补充记录解决此前协议文档中部分数量和路径未决项；此前文档的“未连接远端”仅描述上一轮审计，不适用于本次。

## 1. 数量口径

- 细胞数指当前正式实验保留的配对 RNA–ATAC 细胞/细胞核；同一细胞同时有两种测量，不把 RNA 和 ATAC 相加算成两个细胞。
- RNA 列是基因维度；ATAC 列是 peak 维度，不是 reads、fragments、测序文库数。
- 原生矩阵、选定 HVG、图连接 peak 子集和跨来源 canonical peaks 是不同层次，不能混用。
- donor、技术组、Site 和数据集不是同义词。跨队列生物身份没有全部核验，不能按表中单位数宣称全部互相独立。

## 2. AD：正式 rev04 四组四折

证据：`data/benchmark/AD_FORMAL_DATASET_REGISTRY.csv`、`AD_FORMAL_DONOR_MANIFEST.csv`，以及 `result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04/preprocessing/` 下的 `cohort_summary.csv`、`cohorts/*/evaluation_units.csv`、`native_peak_names.npy` 和原始矩阵头。

| 组 | 数据来源 | 分析单位 | AD/阳性 : 对照/阴性 | 配对细胞数 | 统一 RNA 基因数 | 原生 ATAC peaks |
|---|---|---|---|---:|---:|---:|
| G1 | GSE214979 | 15 donor | 7 : 8 | 105,332 | 36,601 | 150,614 |
| G2 | GSE268609 | 13 技术组/sample group | 8 : 5 | 51,246 | 36,601 | 141,330 |
| G3 | SEA-AD MTG | 24 donor | 21 : 3 | 113,019 | 36,601 | 218,882 |
| G4 | GSE288063 + GSE278576 | 10 donor，11 library | 5 : 5 | 54,999 | 36,601 | 952,335 |
| 合计 | 四个 LODO 组 | 49 donor 记录 + 13 技术组 | 41 : 21 个分析单位 | 324,596 | 不相加 | 不相加 |

G4 的 952,335 是 11 个文库原生 peak 字符串去重后的组内字典，不是最终每折 canonical peak 数。ad22 的两个文库合并为一个 donor。registry 为两个 accession 各保留一行，但该组 10 donor 只能统计一次。G4 的病例/对照与 accession 来源关联，需保留该批次混杂限制。

G2 沿用 orig.ident：1、2、3、4、7、8、26、27、28、30、31、34、36；不能称作 13 个已核验独立生物 donor。G3 阳性为 ADNC Intermediate/High，阴性为 Not AD，排除 Low；这与临床 AD 诊断不是完全相同的终点。

### QC、配对与标签处理

- **G1**：复用作者最终过滤细胞。作者阈值记录为 RNA 检测基因 >200 且 <10,000、percent.mt <5、nucleosome_signal <2、TSS.enrichment >2、dbl <3.5。作者已过滤输入和本地保留均为 105,332，额外删除为 0。原始 counts 与作者预处理后的 RNA QC 统计有差异；报告说明作者曾移除 PMI 相关基因，因此没有拿完整原始 counts 重新套同一 QC。证据：`data/benchmark/AD/GSE214979/author_qc_v1_20261012_bundle/post_qc/QC_REPORT.json`。
- **G2**：保存的原始重建输入 366,175 行，作者子集为 153,530 行；正式 rev04 再按上述 13 组限定为 51,246 行。RNA/ATAC 同步索引，保存整数 counts，不把全作者子集当作正式疾病实验。证据：`data/staging/AD/GSE268609/verified_csr_20261024/manifest.json`，以及 rev04 `provenance/ad_fourfold_prepare.py::source_defs` 中首三组 reader，被 `ad_new_cohort.py::first_three` 复用。
- **G3**：复用作者 QC 后数据，精确对应 RNA 和 ATAC 的配对细胞。二分类筛选前的配对集合为 133,084 细胞；排除 Low 的 4 donor、20,065 细胞，保留 113,019。证据：`result_contain/AD_Dmode_4Cohort_20261024/data_audit/seaad_source3_eligible_counts/manifest.json`；其输入指向 `.../AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout/preprocessing/seaad_paired_counts/manifest.json`。
- **G4**：用作者 metadata 的 **Library + Barcode** 精确选择 11 个 H5 中的细胞，不保留 raw droplets。每个文库检验 RNA 基因顺序一致；组内合并 peak 字典；疾病标签 Normal → Control。证据：rev04 `preprocessing/provenance/ad_new_cohort.py::new_definition` 和 bundle 的 `verification_report_rev04.json`。不能沿用被替换旧 G4 的 RNA-only QC 规则。

### 训练划分

| 来源作为 SOURCE 时 | Encoder train 单位/细胞 | Encoder validation 单位/细胞 |
|---|---:|---:|
| G1 | 12 / 85,038 | 3 / 20,294 |
| G2 | 13 / 51,246 | 0 / 0 |
| G3 | 19 / 81,711 | 5 / 31,308 |
| G4 | 8 / 44,830 | 2 / 10,169 |

G1/G3/G4 按 donor 划分，代码按原始疾病类别分层抽取验证 donor，固定 seed 20020130；G2 没有单独验证组。不能描述成每组都做随机细胞 80/20。来源的这些划分在该来源变为 TARGET 时不用于训练，该折全部 TARGET 细胞仅做冻结变换/推理。SOURCE 疾病标签用于冻结 Encoder 后的 Probe；“不用于表征损失”不等于划分阶段从未读取标签。

### 特征构造与模型流程

实际正式入口是 rev04 `preprocessing/provenance/ad_prepare.py::main`，它复用旧 reader 并用 `ad_new_cohort.py` 替换 G4；不能只看同目录历史 `ad_fourfold_prepare.py` 顶部的旧组名来判断当前队列。

1. Ensembl stable ID 对齐到 36,601 基因，检查无版本后缀且唯一。RNA 每细胞在完整 36,601 基因上 CP10K、log1p，再取选定特征。
2. 每个 SOURCE 仅用其 Encoder train 细胞计算 HVG 分数 variance / max(mean, 1e-6)，每个 donor/技术组总权重相等；每来源 top 2,000，三个 SOURCE 取交集。
3. 代码有交集 <256 时按来源平均秩选 2,000 的回退；**四折实际均未触发**。
4. 只合并该折 SOURCE 的原生 peak 区间；重叠区间合并为 canonical peaks，相邻不重叠区间不合并；原生 peaks 按 genomic overlap 映射，ATAC count >0 二值化。TARGET ATAC 不用于构图或生成 F。
5. GENCODE v32/GRCh38 TSS 候选图：±250 kb、每基因最多 50 个候选 peaks。Stable Regulatory Template 用 SOURCE train 的重复抽样相关性及稳定性筛选；默认 10 次、每次 20%，保留密度 0.5。AD adapter 的来源权重为 Q_d / sum(Q)，不乘来源训练细胞数。
6. `fit_dmode_field`：SOURCE train 拟合 IDF，IDF = log1p(N_train / (1 + df))；TF 为每细胞开放峰深度归一化；TF-IDF 经 B_D 投影到选定基因空间，再按 SOURCE train 均值/标准差标准化，std 加 1e-6。所得 Cell-specific Regulatory Field 与 RNA 有相同基因维度。
7. Teacher 使用 RNA + F；Student 经 MSE 蒸馏，冻结后 TARGET 只输入 RNA，得到 64 维细胞 embedding。
8. donor/技术组内按 Excitatory、Inhibitory、Astrocytes、Oligodendrocytes、OPC 分别取 64 维均值，固定次序拼成 320 维。SOURCE 拟合缺失块均值填补和 StandardScaler；每单位仅一条样本；使用固定 L2 LogisticRegression（C=0.1、lbfgs、max_iter=1000、tol=1e-4、balanced、intercept），阈值 0.5。PCA-RNA64 和 RNA-only Encoder 使用相同 P2 与分类头。

| Held-out TARGET | 实际交集 RNA 基因 | SOURCE canonical ATAC peaks | B_D 调控边 |
|---|---:|---:|---:|
| G1 | 651 | 291,356 | 14,137 |
| G2 | 642 | 272,777 | 13,355 |
| G3 | 661 | 256,234 | 13,731 |
| G4 | 715 | 260,503 | 14,341 |

证据：rev04 `preprocessing/preprocessing_summary.csv`、`folds/fold*/hvg_audit.json`、`selector_manifest.json`；上表是预处理完成状态，不把其中历史 gpu_training_started=False 字段当作当前训练进度。

## 3. PBMC：四来源及新旧数据区别

证据：`data/protocols/PBMC/corrected_legacy_evidence/UPDATED_DATASET_PROVENANCE_TABLE.tsv`、`UPDATED_PBMC002_MANIFEST.json`、各来源 `data/benchmark/PBMC/PBMC-MO-00*/legacy_corrected_refs_v1/native_access/` 的身份文件和当前矩阵，以及 corrected S/D cache。

| 来源 | 正式身份 | 发布来源 donor 描述 | 配对细胞 | 完整 RNA 字典 | 原生 ATAC peaks |
|---|---|---|---:|---:|---:|
| D1 | 10x granulocyte-sorted 10k，ARC1.0.0 | 一名健康女性、25 岁 | 9,347 | 36,601 | 108,377 |
| D2 | GSE196990，HDa（RNA GSE196988 + ATAC GSE196987） | 一名健康女性、36 岁 | 6,317 | 36,601 | 212,886 |
| D3 | 10x unsorted 10k，ARC2.0.0 | 一名健康女性、25 岁 | 8,105 | 36,601 | 111,857 |
| D4 | 10x unsorted 3k，ARC2.0.0 | 一名健康女性、25 岁 | 2,413 | 36,601 | 81,156 |
| 合计 | 四个数据发布/实验来源 | 跨发布独立 donor 总数未核实 | 26,182 | 不相加 | 不相加 |

来源表有每个发布 one-donor 的描述，但 D1/D3/D4 缺乏可跨发布核对的生物身份，不能据此宣称“四个独立 donor”。旧 identity JSON 对 donor 写 not specified，与后来来源表补充的描述属于不同证据粒度。表达矩阵 exact-count fingerprint 无重复不等于证明不同 donor。

### QC 与当前 scReLIFT-S

- 当前 cohort 和细胞名单沿用冻结处理结果，不能把 `legacy_corrected_refs_v1` 当作一次新 QC；其 manifest 明示不是完整 post-QC H5AD 包。
- D2 从作者 post-QC paired cohort 选择 HDa，按完整 cell ID 对应 RNA、ATAC 和标签。
- D1 H5 文件名含 raw，包含 736,320 个原始 barcode；这是 droplets/barcodes，不是本项目最终细胞数。D3/D4 官方 filtered H5 分别有 12,012 / 3,009 行，正式名单为 8,105 / 2,413。原始 H5 行数不能直接替换项目 cohort 数。
- 检查到的历史 PBMC QC 函数为 RNA total 与 ATAC open peaks 各 P1–P99.5，另一个 split 函数移除少于 10 细胞的类别；位置 `V5/v5_final_20260906/diagnostics/prepare_pbmc10k_multiome.py::filter_cells/split_labels`。**本次未把每个来源最终名单反向完整重放到此函数，不把这一历史函数宣称为四来源统一 QC。**
- 当前 corrected scReLIFT-S cache 独立来源选择 2,000 HVG：全库 CP10K/log1p、仅训练细胞 variance/mean 排序，候选池为全部验证基因，不先删除不连图基因。保留该来源原生 peak 空间，SOURCE train 建 B_S、拟合 TF-IDF/F 和 RNA 标准化。测试为 RNA-only。
- 证据：`data/cache/s_mode/PBMC/PBMC-MO-00*/pbmc_corrected_s10x20_v1_20261012/manifest.json`；共用实现 `main/core/regulatory/s_mode/single_source.py::normalize_counts/select_hvg/build_bs/build_field`。

| 来源 | Train | Validation | Test | 当前 scReLIFT-S RNA / ATAC 维度 |
|---|---:|---:|---:|---|
| D1 | 6,542 | 1,402 | 1,403 | 2,000 / 108,377 |
| D2 | 4,421 | 948 | 948 | 2,000 / 212,886 |
| D3 | 5,677 | 1,214 | 1,214 | 2,000 / 111,857 |
| D4 | 1,693 | 360 | 360 | 2,000 / 81,156 |

这是来源内细胞划分，不能称 donor-held-out。Intra 用本来源 Test；Inter 将冻结 RNA-only Student 应用到其他来源，沿用 SOURCE 特征及变换。

### 已审计 corrected scReLIFT-D cache

`main/run/d_mode/prepare_updated_b2_sources.py::main` 与 `main/core/regulatory/common/legacy_support/v5d/harmonize.py::prepare_harmonized` 复用每来源旧 prepared top-2,000 panel，取三个 SOURCE panel 的交集，再对 **prepared peak 子集** 做区间合并并集。因此不能把这些 peak 数说成完整原生 peak 并集，也不能把此旧 prepared 路径等同于当前 corrected scReLIFT-S 全基因候选 HVG 路径。旧 HVG 函数在 graph-connected candidate 子矩阵内先归一化，其最终 RNA 保存又是全库归一化；不应把 AD 的全库 HVG 实现照搬描述成这份旧 PBMC 实现。

| Held-out TARGET | RNA 交集基因 | SOURCE canonical peaks |
|---|---:|---:|
| D1 | 328 | 94,228 |
| D2 | 449 | 67,669 |
| D3 | 339 | 90,634 |
| D4 | 370 | 92,331 |

prepared SOURCE peak 子集 D1/D2/D3/D4 分别为 27,900 / 58,347 / 38,640 / 32,293。证据为 `data/cache/d_mode/PBMC/legacy_full_b2_corrected/target_D*/source_artifacts/harmonized/gene_harmonization_manifest.json` 及实际保存的 gene/peak 数组。其 held_out_target_dataset=null、target_excluded_from_feature_universe=false 是调用未传该参数的元数据限制；实际 feature_reference_datasets 与 datasets 列表只有三个 SOURCE，不能单凭此布尔字段断言 TARGET 参与了特征拟合。

**重大路径陷阱**：`data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D2/` 仍含旧 D2，RNA 为 10,137 × 36,601，ATAC 为 10,137 × 143,887；它不是当前 6,317 细胞的 HDa。未删除历史文件，当前统计使用 corrected 身份清单和 native_access。

## 4. BMMC：10 donor、四 Site

证据：`data/raw/BMMC/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad` 的矩阵头，`result_contain/96_bmmc_multiome_recovery/{MODALITY_AUDIT.json,SITE_CELL_COUNTS.csv,SITE_DONOR_DISTRIBUTION.csv}`。

- 69,249 配对细胞；13,431 RNA 基因；116,490 ATAC peaks；联合 AnnData 形状 69,249 × 129,921；22 个作者细胞类型。
- 10 个 DonorID；13 个 Site–donor 组合；4 个 Site。DonorID 15078 在 Site1/2/4，18303 在 Site1/3 出现，不能将各 Site donor 数直接相加当作独立 donor。
- 读取 `layers/counts` 的原始整数 counts，复用作者处理后细胞和注释；本地恢复审计没有重新删细胞或重新注释。不能把作者 upstream QC 阈值臆造为本地阈值。

| Site | 细胞 | donor 数 | 类型数 | Train / Validation / Test |
|---|---:|---:|---:|---|
| Site1 | 17,243 | 3 | 22 | 12,070 / 2,586 / 2,587 |
| Site2 | 15,226 | 3 | 20 | 10,658 / 2,284 / 2,284 |
| Site3 | 14,556 | 4 | 21 | 10,189 / 2,183 / 2,184 |
| Site4 | 22,224 | 3 | 20 | 15,556 / 3,334 / 3,334 |

实现 `data/run/s_mode/prepare_bmmc_site_intra.py::load/prepare`：按 Site 提取，按作者 cell_type 分层做 70/15/15 细胞划分，固定 seed 20020130。各 Site 训练细胞独立 CP10K/log1p、选 2,000 HVG、拟合 RNA mean/std；ATAC 二值化、保留 116,490 原生 peaks；使用 GRCh38/GENCODE v32、±250 kb、每基因最多 50 peaks 构建候选图，再用共用 `build_bs/build_field` 得到 Site-specific B_S 与 2,000 维 F。F 对 train/val 生成，测试 ATAC 不进入 F；冻结 Student 用测试 RNA 推理。输出保存在 `data/cache/s_mode/BMMC/bmmc_site_intra_v1_20261024/BMMC_Site*/`。

这是 **四 Site Intra**，不是四 donor 分别留出，也不是跨 donor 验证；本次不将历史 Site-to-Site Inter 混入当前 Intra 数量。

## 5. 本次新增确认与仍有限制

新增确认：G1 为 7 AD +8 Ctrl；AD 四折实际 HVG/peak/edge 数；G2 无独立验证组；当前 PBMC D2 与退休 staging 文件不同；PBMC 当前 scReLIFT-S 全原生 peaks 和旧 scReLIFT-D prepared 子集不同；BMMC 10 donor/13 Site–donor 的对应关系。

仍未独立重建：PBMC 每来源最终 QC 全链条与每条删细胞原因、跨发布 donor 身份、G2 生物 donor 映射；作者 upstream QC 的所有外部细节；本次扫描未重跑训练或完整原始数据哈希审计。SEA-AD 最小 donor 仅 3 个配对细胞，名单证据保留，不能凭总细胞数假设所有 donor 的五类覆盖充分。

服务器 AGENTS.md 当前种子规定为 20020130、20020131、20020132，报告 mean ± sample SD (ddof=1)；五/十五 seeds 仅可作为历史证据。

