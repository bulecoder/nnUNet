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
* windows上面训练：在终端中使用命令`set NNUNET_n_proc_DA=6`和`set NNUNET_COMPILE=0`（windows下禁用编译优化，避免CUDA编译错误），使用`nnUNetv2_train 501 3d_fullres 1`启动训练
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
* 在windows中训练完成以后进行validatio的时候，由于CBCT数据太大可能会爆显存，导致validation终端，可以将数据移到服务器上面以后再进行validation，使用命令`CUDA_VISIBLE_DEVICES=1 NNUNET_COMPILE=f nnUNet_n_proc_DA=6 nnUNetv2_train 507 3d_fullres 2 --val` --val表示不要训练，直接启动验证，-disable_postprocessing_on_folds表示不用进行自带的LCC后处理，这个后处理之后尝试判断需不需要LCC后处理且如果需要只会生成postprocessing.json文件，不会修改validation文件夹中的内容，我们不需要自带后处理；


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
TUV        | 1
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


* 使用LCC后处理以后，整体结果变好了，但是也有一些样本的结果变差了，而且这些问题主要集中在AICV和BEP这两个点上面


### 下一步建议

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


实验记录：
* 筛选出包含AICV、BEP和TEE的样本，但是不屏蔽其他标签进行训练得到的结果（503任务）不理想，4个fold发生了数据坍塌，删除503任务，得出结果数据处理的时候半径可能太小了；
* 筛选出包含AICV、BEP和TEE的样本，屏蔽其他标签进行三分类训练（504任务），结果优于503任务；
* 501任务已经确定了半径为3，太小了，效果不好，直接删除，全面被505（半径设置为6）替代；
* 504也没有意义了，也是半径为3，筛选出标签1、3、5都存在的样本屏蔽其他标签，映射为三分类；后续需要覆盖为半径为6的三分类；
* 505的数据预处理（prepareData_landmark.py）有问题，没有加入正则化匹配，如果患者姓名里面包含了关键点的缩写，会导致label标识错误，也没有严格后缀匹配，可能导致关键点丢失（但是这一版本结果还比较好）；
* 507使用新的数据预处理脚本（prepareData_landmark2.py），同样使用半径为6进行划分；
* 502任务的原始数据中有46个数据标注有问题，结果很差，平均dice在0.6左右，好的结果在0.9以上查的结果小于0.1，502无效
* 506任务是修正后的数据，确保分割标注数据全部正确;
* 508任务以半径为6，确保数据标注完全对应正确的情况下，筛选出三个点进行训练，效果没有提升，完全失败的尝试;
* 509任务以半径为8进行划分，相比于半径为6的时候误差并没有减少，相反有些还增加了
* 510任务以半径为6进行划分，MALOUIN_JACQUELINE_1932_11_10这个样本的label专门由prepareData_seg2.py来处理。510的结果全面比506要更好，更强的鲁棒性，更优的边界质量；
* 511任务，纠正了几个原始定点标注错误，然后使用prepareData_landmark2.py来进行数据预处理（保留患者姓名）。511的结果全面比507要更好，平均误差降低一点点，标准差全面降低;


数据检查记录：
* MALOUIN_JACQUELINE_1932_11_10这个样本，CBCT和STL是对齐的，但是使用prepareData_seg.py脚本转换为nii.gz以后，就不对齐了，使用prepareData_seg2.py来单独处理这个样本，数据保存在test_alignment文件夹中（定点和CBCT是对齐的）

* KARILAID_Tarmo 这个样本，定点的nii.gz和分割的image放在一起，定点的nii.gz是椭圆形不是圆形，但是点的位置是对的；定点的nii.gz和定点的image放一起，同样是对齐的，但是nii.gz是椭圆形不是圆形

已解决：
* STEPHANIS_James_1958_6_30 这个样本定点的标注和定点的CBCT图像不对齐
* STUCKGOLD_ANDREW 这个样本定点的标注和定点的CBCT图像不对齐
* THOMPSON_Gerard_1949_5_3 这个样本定点的标注和定点的CBCT图像不对齐


## 分割
* 使用prepareData_seg.py以后，得到的label和image是对齐的，但是label和stl是不对齐的，也就是说事实上stl和image也是不对齐的。这是因为STL在导出的时候缺少了坐标系的信息，而在做预处理的时候实际上是根据 CBCT 的“地图”把 STL 重新定位并画在了正确的位置上。
* 后续人工校对以后，发现有46个样本数据有问题，由stl得到的label也无法与image对齐，返工重新标注完成，现在所有由stl得到的label都可以与image对齐;
