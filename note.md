# 操作说明

### 环境配置
```
conda create -n nnunetv2 python=3.10 -y
conda activate nnunetv2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2

# 配置conda环境变量，根据具体情况修改路径名字
conda env config vars set nnUNet_raw=F:\nnUNet\nnUNet_raw
conda env config vars set nnUNet_preprocessed=F:\nnUNet\nnUNet_preprocessed
conda env config vars set nnUNet_results=F:\nnUNet\nnUNet_results
# 配置完成以后重新激活conda环境
# 验证是否成功（windows的命令）
echo %nnUNet_raw%
echo %nnUNet_preprocessed%
echo %nnUNet_results%
# linux命令
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

目录结构例子：
```
nnUNet_raw/
└── Dataset001_AirwayLandmark/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json

nnUNet_preprocessed/
└── Dataset001_AirwayLandmark/
    ├── nnUNetPlans.json
    ├── dataset_fingerprint.json
    └── ...

nnUNet_results/
└── Dataset001_AirwayLandmark/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        ├── fold_0/
        ├── fold_1/
        └── checkpoints/

```

* 先运行 `prepareData.py`进行数据预处理，数据会处理到 `nnUNet_raw`文件夹中
* 再运行 `nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -npfp 1 -np 1`（单进程处理，windows下多进程容易出错），数据会处理到 `nnUNet_preprocessed`中。该命令会默认处理为nnUNetPlans_2d、nnUNetPlans_3d_fullres、nnUNetPlans_3d_lowres，如果只需要nnUNetPlans_3d_fullres，可以使用`nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity`
* 开始训练，`nnUNetv2_train 501 3d_fullres 0`启动训练，501为任务ID，直接使用3d_fullres配置，0代表Fold 0（默认有5个Fold），默认打开12个以上的进程，在windows上运行容易出错；`CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 501 3d_fullres 1`指定显卡进行训练;`CUDA_VISIBLE_DEVICES=1 NNUNET_COMPILE=f nnUNetv2_train 501 3d_fullres 1`linux下会自动开启编译模式，但是看起来并没有更快，关闭编译的命令；`CUDA_VISIBLE_DEVICES=1 NNUNET_COMPILE=f nnUNet_n_proc_DA=6 nnUNetv2_train 501 3d_fullres 1`linux上面是机械硬盘，默认开启12个进程会导致IO颠簸，降低进程数量（目前看起来6个进程比较合适）；
    * 设置环境变量，设置使用主进程加载数据；忽略libiomp 和 libomp 冲突的红字警告；（临时生效，每次打开终端都要输一次）
    ```
    set nnUNet_n_proc_DA=0
    set KMP_DUPLICATE_LIB_OK=TRUE
    ```
* 监控训练：`nnUNet_results/Dataset501_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/`路径下的文件，打开`progress.png`，图片会实时更新  
    * 蓝色线 (Train Loss)：应该持续下降。
    * 红色线 (Validation Loss)：应该持续下降。
    * 绿色线 (Pseudo Dice)：这是最关键的指标，代表模型在验证集上的分割准确率，应该持续上升。
* 训练中断以后，使用`nnUNetv2_train 501 3d_fullres 0 --c`断电续训，自动读取最新的 checkpoint继续跑
* 验证。在训练完成以后会自动开始验证，但是验证集中数据太大，在windows中自动开启多个进程来处理和保存结果时内容爆了，导致RuntimeError: Some background workers are no longer alive；补跑验证（指定显卡）：`CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 501 3d_fullres 0 --val`，在windows下多进程机制在结束进程后，内存回收和句柄释放有延迟，导致内存(RAM)或进程句柄爆了；直接用模型进行预测（推理）：`nnUNetv2_predict -i <你的输入文件夹> -o <你的输出文件夹> -d 501 -c 3d_fullres -f 0`；不仅要预测结果，还要看验证病例的分数：`nnUNetv2_evaluate_folder -gt "F:\CBCT\nnUNet\nnunetv2\nnUNet_raw\Dataset501_AirwayLandmarks\labelsTr" -pred "F:\CBCT\inference_output" -djfile "F:\CBCT\nnUNet\nnunetv2\nnUNet_preprocessed\Dataset501_AirwayLandmarks\dataset.json"`
    ```
    # 评估命令
    nnUNetv2_evaluate_folder \
    -gt "nnUNet_raw/Dataset501_AirwayLandmarks/labelsTr" \
    -pred "nnUNet_results/Dataset501_AirwayLandmarks/fold_0/validation" \
    -djfile "nnUNet_preprocessed/Dataset501_AirwayLandmarks/dataset.json"
    ```
    npp：将负责预处理的进程数限制为1，nps将负责导出分割结果的进程数限制为1;
    ```
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict \
    -i "nnUNet_raw/Dataset501_AirwayLandmarks/imagesTr" \
    -o "nnUNet_results/Dataset501_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation2" \
    -d 501 \
    -c 3d_fullres \
    -f 0 \
    -nps 1 \
    -npp 1
    ```

* 训练完成以后，如果不报错会自动生成一个summary.json文件，progress.png、以及validation里面的预测结果


缺少hiddenlayer库，尝试安装一下



### 数据分布
==============================
点数(有效STL)       | 病例个数
------------------------------
4               | 4
5               | 43
6               | 71
7               | 132
------------------------------

==============================
 📉 各关键点缺失详情
==============================
关键点名称      | 缺失病例数
------------------------------
AICV       | 68
ANS        | 4
BEP        | 20
PNS        | 1
TEE        | 73
TEP        | 2
TUV        | 4
------------------------------


### 结果评估
关键点分级诊断 (Landmark Tiers)
根据这 5 个 Fold 的表现，我们可以把这 7 个点分为四个梯队：

🏆 第一梯队（发挥稳定且精准）：PNS, TEP, TUV
表现：这三个点在所有的 Fold 中都没有出现大面积漏检，且 MRE 稳定在 2.5mm - 5mm 之间。SDR@4mm 基本都在 75% 以上。
结论：这说明这三个点在 CBCT 图像上的解剖特征非常明显，且你的标注一致性很好，模型已经基本“学会”了它们。

🥈 第二梯队（极具潜力但遇过“车祸”）：ANS
表现：ANS 在 Fold 1, 2, 3, 4 中表现堪称完美（Fold 4 的 MRE 甚至达到了惊人的 1.545mm，SDR@4mm 逼近 96%！）。但是，它在 Fold 0 中出现了0匹配、49漏检的惨剧。
结论：Fold 0 中 ANS 通道的网络权重大概率“死掉”了。这可能是由于 Fold 0 的训练集中刚好集中了太多没有标注 ANS 的残缺数据，导致网络为了降低 Loss，直接把预测 ANS 的概率全部置零了。

⚠️ 第三梯队（能找到但找不准）：BEP, TEE
表现：除了 Fold 2 全部漏检外，其他 Fold 都能找出一部分，但是误差极大（MRE 在 7mm - 14mm 徘徊），SDR@2mm 极低。
结论：模型知道这俩点大概在哪个区域，但无法精确定位。这通常是因为这两个点在不同病人身上的形态变异太大，或者医生标注时的标准不够统一。

🆘 第四梯队（彻底的灾难）：AICV
表现：Fold 0, 1, 2, 3 全部为 0 匹配（全部漏检）。到了 Fold 4 突然“诈尸”匹配了 32 个，但 MRE 高达 10mm，且伴随 10 次误检（无中生有）。
结论：模型根本没有学会 AICV 的特征。

这是 nnU-Net 默认损失函数 (Loss Function) 带来的副作用：

nnU-Net 默认使用 CrossEntropy + Dice Loss。

当你的训练数据中，某一张 CBCT 确实存在 AICV 这个解剖结构，但医生忘记标注或者因为视野原因没标时，由于没有标签，这个区域会被当做**“背景 (Background)”**。

如果模型极其聪明，通过周围组织的上下文找出了 AICV 并给出了预测，此时计算 Loss 时，模型会因为“预测了背景”而受到严厉的惩罚。

经过几百个 Epoch 后，模型学到了一个“保命策略”：既然预测 AICV 这么容易挨罚，那我就干脆永远不预测 AICV（全部输出 0）。这就是为什么你在大部分 Fold 里看到 AICV 有 30 多例漏检。

Fold 4 中的 AICV 出现了 10 次“误检(GT无预测有)”，这恰恰证明了模型其实找到了它，只是你的 GT 里没有标，导致它被扣上了“误检”的帽子。


### 下一步建议
修改网络策略（进阶）
如果要彻底解决“未标注点被当成背景惩罚”的问题，你可能需要放弃纯净版 nnU-Net 的一键运行，去修改它的代码：
屏蔽梯度（Masked Loss）：在计算 Loss 的代码里加一个 Mask，如果某个病例 GT 里没有标 AICV，那么模型在 AICV 这个通道上的预测结果就不参与 Loss 计算（既不奖励，也不惩罚）。


方案一：数据分流训练法（Sub-dataset Training） —— 最推荐，改动代价为 0，见效最快
既然 nnU-Net 是个“死脑筋”，我们就不要强迫它在一个模型里学完所有东西。你可以把任务拆分成两个独立的模型：
模型 A（专注核心点）：
数据：使用全部 250 例 数据。
标签：只保留 PNS, TEP, TUV, ANS 这 4 个（或者加上 BEP 凑 5 个）几乎不缺失的点。把 AICV 和 TEE 的标签抹掉。
预期：这 4 个点的精度会进一步逼近人类极限。
模型 B（专攻困难点）：
数据：只提取那 132 例 拥有完整 7 个点标注的“完美病例”。
标签：保留全部 7 个点。
预期：在没有任何“错误惩罚”的纯净数据下，AICV 和 TEE 的表现会瞬间起飞，脱离 0 匹配的泥潭。

方案二：修改损失函数（Masked Loss） —— 算法大佬路线，工程难度极大
如果你不想碰数据，想纯靠算法解决“部分标签缺失 (Partially Labeled Data)”的问题：
原理：修改 nnU-Net 的底层 Python 代码。在计算 Loss 之前加一行判定：如果当前病例的 GT 里没有 Label 1 (AICV)，那么直接把 Label 1 的预测梯度截断（设为 0），不奖励也不惩罚。


实验记录：
* 筛选出包含AICV、BEP和TEE的样本，但是不屏蔽其他标签进行训练得到的结果（503任务）不理想，4个fold发生了数据坍塌，删除503任务，得出结果数据处理的时候半径可能太小了；
* 筛选出包含AICV、BEP和TEE的样本，屏蔽其他标签进行三分类训练（504任务），结果优于503任务；



## 分割
* 使用prepareData_seg.py以后，得到的label和image是对齐的，但是label和stl是不对齐的，也就是说事实上stl和image也是不对齐的。这是因为STL在导出的时候缺少了坐标系的信息，而在做预处理的时候实际上是根据 CBCT 的“地图”把 STL 重新定位并画在了正确的位置上。