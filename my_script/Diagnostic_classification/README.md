# 非 OSA 对照组：推理与后处理

本目录沿用原有 nnU-Net 流程，只新增一个面向外部对照组的参数化后处理脚本。

- 推理：直接使用原生 `nnUNetv2_predict`，不再包装。
- 后处理：使用 `postprocess_control_group.py`。
- 指标：预测与后处理确认无误后，再单独适配原有指标脚本。

## 独立路径

```text
输入：/data1/xyh/data/control_group

输出：/data1/xyh/data/control_group_output
├── airway_510_raw
├── airway_510_lcc
├── landmarks_511_raw
└── landmarks_511_lcc

脚本和报告：
/data1/xyh/projects/nnUNet/my_script/Diagnostic_classification
├── qc_airway_510.py
├── postprocess_control_group.py
└── reports
```

这些路径不会写入旧的 `nnUNet_results/fold_*/validation` 或旧 OSA metrics 目录。

## 1. 环境

```bash
source /data1/xyh/miniconda3/etc/profile.d/conda.sh
conda activate nnunetv2

export nnUNet_raw=/data1/xyh/data/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/data1/xyh/data/nnUNet/nnUNet_preprocessed
export nnUNet_results=/data1/xyh/data/nnUNet/nnUNet_results

INPUT=/data1/xyh/data/control_group
OUTPUT=/data1/xyh/data/control_group_output
```

输入目录应有41个 `CaseXXX_0000.nii.gz`；Case006和Case029缺失是预期。

## 2. 原生510分割推理

```bash
CUDA_VISIBLE_DEVICES=0 NNUNET_COMPILE=f nnUNetv2_predict \
  -i "$INPUT" \
  -o "$OUTPUT/airway_510_raw" \
  -d Dataset510_AirwaySegmentation \
  -c 3d_fullres \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  -f 0 1 2 3 4 \
  -chk checkpoint_final.pth \
  -npp 1 \
  -nps 1
```

## 3. 原生511定点推理

可在另一张空闲GPU的独立终端或tmux会话中运行：

```bash
CUDA_VISIBLE_DEVICES=1 NNUNET_COMPILE=f nnUNetv2_predict \
  -i "$INPUT" \
  -o "$OUTPUT/landmarks_511_raw" \
  -d Dataset511_AirwayLandmarks \
  -c 3d_fullres \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  -f 0 1 2 3 4 \
  -chk checkpoint_final.pth \
  -npp 1 \
  -nps 1
```

两项预测均保留默认TTA；不要添加 `--disable_tta`，也不需要
`--save_probabilities`。输出文件会命名为 `CaseXXX.nii.gz`。

若预测中断，确认旧进程已经退出、输入和所有模型参数完全未变后，可在原命令末尾追加：

```text
--continue_prediction
```

## 4. 510原始分割QC

在后处理前先运行只读QC。脚本不会修改任何NIfTI，只会在
`reports/airway_510_qc` 下生成逐病例CSV、异常病例Markdown和汇总JSON：

```bash
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/qc_airway_510.py
```

QC重点检查连通分量、LCC潜在删除比例、三维封闭空洞、index-z方向的内部空层和
局部面积骤降、图像边界接触、几何一致性及队列容积离群。index-z平面面积仅用于
筛查，不等同于后续沿气道方向计算的CSAmin。先阅读异常病例报告并复核标记病例，
再进行后处理。

## 5. 后处理

只有相应 raw 输出目录完整包含41例后才运行：

```bash
POST=/data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/postprocess_control_group.py

python "$POST" --task 510
python "$POST" --task 511
```

- 510：保留最大物理体积的气道连通域。
- 511：逐标签保留最大连通域质心，并严格复刻旧流程的
  `6 / mean(spacing)` 后取整数体素半径的重绘方式。
- 脚本检查41例Case集合、标签值和输入/输出几何，并记录空预测、缺失定点、边界裁切、
  球体覆盖冲突和质心偏移。
- 报告写入本目录的 `reports` 文件夹。

后处理目录非空时脚本默认拒绝运行。只有在确认要完整重建时才使用：

```bash
python "$POST" --task 510 --overwrite
python "$POST" --task 511 --overwrite
```

完成后四个输出目录都应各有41个 `CaseXXX.nii.gz`。Case012和Case028仍需要人工检查
矢状位视野是否完整覆盖上气道及所需定点。511原始预测中Case012缺少AICV、Case020
缺少TUV，详见 `landmarks_511_raw/landmark_missing_cases.md`。确认预测和后处理结果后，
再继续编写/适配本目录下的指标计算代码。
