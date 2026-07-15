# OSA 组现有指标覆盖审计报告

- 生成日期：2026-07-14
- OSA 原始 CBCT 病例数：250
- 原始手工定点病例目录数：250
- 原始手工气道分割 STL 数：240
- 指标结果目录：`/data1/xyh/projects/nnUNet/my_script/evaluation/metrics`

## 1. 六项指标覆盖情况

| 编号 | 指标 | 有效病例数 | 覆盖率 | 主要限制 |
|---|---|---:|---:|---|
| 1 | 上气道容积 | 240 / 250 | 96.0% | Dataset510 和手工气道 STL 仅覆盖 240 例 |
| 2 | mCSA | 235 / 250 | 94.0% | 10 例无气道分割；另有 5 例无法由手工 ANS/PNS 生成头位校正参数 |
| 3 | ANS–PNS 长度 | 250 / 250 | 100.0% | 当前结果来自 Dataset511 五折验证预测及 LCC 后处理 |
| 4 | BEP–TEE 长度 | 176 / 250 | 70.4% | 71 例预测缺 TEE，3 例预测缺 BEP |
| 5 | PNS–TUV 长度及软腭高度 | 250 / 250 | 100.0% | 当前结果来自 Dataset511 五折验证预测及 LCC 后处理 |
| 6 | PNS–TUV–BEP–TEE 四点面积 | 176 / 250 | 70.4% | 与第 4 项为同一组缺失：71 例缺 TEE，3 例缺 BEP |

六项指标同时完整的病例为 **163 / 250（65.2%）**；共有 87 例至少缺少一项指标。

> 注意：`2.mcas.csv` 共 236 行结果记录，其中 1 行为 `TOTAL_AVERAGE` 汇总行，因此实际患者结果为 235 例，不能把汇总行计为患者。

## 2. mCSA 额外缺失的 5 例

这 5 例均存在 Dataset510 分割标签、GT STL 和验证预测 STL，但 `rotation_params_511.txt` 中被标记为 `MISSING`，原因是原始手工定点缺少 ANS 或 PNS，无法计算基于 ANS–PNS 的头位校正参数。

| 病例 | 缺失手工定点 |
|---|---|
| `BROWN_Eleanor_1946_11_2` | ANS |
| `Beauvais_Germain_1947_8_11` | ANS |
| `LAPOINTE_Gwen_1943_7_11` | ANS |
| `WQP 6` | PNS |
| `ZENGHE NM` | ANS |

因此 mCSA 的有效病例计算为：240 个有气道分割的病例，减去上述 5 个缺少旋转参数的病例，得到 **235 例**。

## 3. 缺少手工气道分割 STL 的 10 例

下列病例存在原始 CBCT 目录，但 `/data1/xyh/data/software_analysis2/stl` 中没有对应气道 STL，因此未进入 Dataset510，现有容积和 mCSA 流程无法覆盖：

1. `ARMSTRONG_Ronald_1936_8_7`
2. `BEY_Fadila_1946_12_17`
3. `COLAVITTI_Edmond_1967_2_18`
4. `FINNEGAN_Dominic_1970_7_8`
5. `GOBRAN_Shaheer_1956_8_1`
6. `Mayette_Jeanine_1947_6_8`
7. `RICHTER_Michael_1949_7_25`
8. `SANGTHONG_Anuchit_1970_10_10`
9. `Tamer_A._1967_9_1`
10. `WILSON_Alastair_1969_8_19`

## 4. 原始手工定点覆盖

| 定点 | 有标注病例数 | 缺失病例数 |
|---|---:|---:|
| AICV | 182 | 68 |
| ANS | 246 | 4 |
| BEP | 230 | 20 |
| PNS | 249 | 1 |
| TEE | 177 | 73 |
| TEP | 248 | 2 |
| TUV | 249 | 1 |

组合覆盖情况：

- ANS 与 PNS 同时存在：245 / 250。
- ANS、PNS、TUV 同时存在：244 / 250。
- PNS、TUV、BEP、TEE 四点同时存在：158 / 250。

这说明“手工定点流程已经完成”不等于每个病例都具有全部七种定点；TEE 是限制 BEP–TEE 和四点面积覆盖率的主要因素。

## 5. 当前结果文件

- `1.volumes.csv`：240 个患者结果。
- `2.mcas.csv`：235 个患者结果 + 1 行 `TOTAL_AVERAGE`。
- `3.ans_pns_length.csv`：250 个患者结果。
- `4.bep_tee_length.csv`：176 个成功结果，74 个缺失结果。
- `5.pns_tuv_and_height.csv`：250 个患者结果。
- `6.pns_tuv_bep_tee_tissue_area.csv`：176 个成功结果，74 个缺失结果。

## 6. 当前结论

OSA 组的六项指标尚未全部计算完成。现有结果可用于审计和方法验证，但若分类模型要求完整六项输入，目前只有 163 例可直接使用。缺失值处理、补标或重新推理应在后续分类建模方案中预先定义，不能把 `TOTAL_AVERAGE` 当作患者，也不应在不评估选择偏倚的情况下直接删除所有缺失病例。