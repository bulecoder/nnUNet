# 非 OSA 对照组指标计算

本目录沿用 `evaluation/metrics` 的编号、独立脚本和结果文件命名。脚本默认读取已经完成的 Task 510/511 后处理结果，CSV 也写入本目录。六项脚本彼此独立，不会修改任何 NIfTI 文件。

## 输入

- 气道分割：`/data1/xyh/data/control_group_output/airway_510_lcc`
- 定点结果：`/data1/xyh/data/control_group_output/landmarks_511_lcc`
- 510 QC：`../reports/airway_510_lcc_qc/airway_510_qc_all_cases.csv`
- 511 QC：`../reports/postprocess_task511.csv`

两个输入目录当前应对应同一组 41 个 `CaseXXX` 病例，即 `Case001`–`Case043` 中排除 `Case006` 和 `Case029`。脚本会核对精确病例集合，避免对不完整或混入错误病例的队列生成结果。

## 脚本与结果

| 编号 | 脚本 | 默认结果 |
|---|---|---|
| 1 | `1.calculate_volume.py` | `1.volumes.csv` |
| 2 | `2.calculate_mcsa.py` | `2.mcas.csv` |
| 3 | `3.calculate_ANS_PNS.py` | `3.ans_pns_length.csv` |
| 4 | `4.calculate_BEP_TEE.py` | `4.bep_tee_length.csv` |
| 5 | `5.calculate_PNS_TUV.py` | `5.pns_tuv_and_height.csv` |
| 6 | `6.calculate_PNS_TUV_BEP_TEE.PY` | `6.pns_tuv_bep_tee_tissue_area.csv` |

所有 CSV 都保留 41 行病例。不能计算的值写为 `NA`，不会用 0 代替；`Status` 记录计算状态，`QC_Flags` 继承已有后处理/QC警告，第 2 项会合并 510 与 511 警告。`2.mcas.csv` 的拼写特意沿用旧 `evaluation/metrics` 结果名。

## 定义保持不变

- Volume：Task 510 LCC 前景体素数乘以 NIfTI 物理体素体积，单位同时输出 `mm³` 和 `cm³/mL`。
- mCSA：保持旧 OSA 流程，即 marching cubes、10 次 Laplacian 平滑、STL 序列化、按 ANS–PNS 旋转、沿旋转后 z 轴在 10%–90% 范围以 0.5 mm 扫描、过滤不大于 5 mm² 的截面，并取 5 层滑动平均后的最小面积。旋转参数由本批 511 后处理定点逐病例重新计算，不会错误复用 OSA 组的手工定点参数。
- ANS–PNS、BEP–TEE、PNS–TUV：使用 SimpleITK 物理质心计算三维欧氏距离。
- Soft palate height：保持旧代码定义，为 TUV 到三维 ANS–PNS 直线的垂直距离。
- 四点面积：保持旧代码定义，为三维三角形 `PNS-TUV-BEP` 与 `PNS-BEP-TEE` 面积之和。

## 已知缺失与预期覆盖

- `Case020` 缺少 TUV，因此第 5 项的两个数值和第 6 项面积应为 `NA`。
- `Case012` 缺少 AICV，但 AICV 不参与本目录六项指标。
- 其他已记录异常保持当前结果并写入 `QC_Flags`，不自动剔除。
- 若计算过程没有额外失败，预期第 1、2、3、4 项有效 41 例，第 5、6 项有效 40 例。

## 手动运行

激活服务器的 `nnunetv2` 环境后逐项执行：

```bash
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/metrics/1.calculate_volume.py
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/metrics/2.calculate_mcsa.py
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/metrics/3.calculate_ANS_PNS.py
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/metrics/4.calculate_BEP_TEE.py
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/metrics/5.calculate_PNS_TUV.py
python /data1/xyh/projects/nnUNet/my_script/Diagnostic_classification/metrics/6.calculate_PNS_TUV_BEP_TEE.PY
```

这些脚本均为 CPU 计算，不需要指定 `CUDA_VISIBLE_DEVICES`。第 2 项需要环境中已有的 `trimesh` 和 `scikit-image`，通常会明显慢于其他五项。
