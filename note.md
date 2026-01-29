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
* 再运行 `nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -npfp 1 -np 1`（单进程处理，windows下多进程容易出错），数据会处理到 `nnUNet_preprocessed`中
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
* 训练终端以后，使用`nnUNetv2_train 501 3d_fullres 0 --c`断电续训，自动读取最新的 checkpoint继续跑
* 验证。在训练完成以后会自动开始验证，但是验证集中数据太大，在windows中自动开启多个进程来处理和保存结果时内容爆了，导致RuntimeError: Some background workers are no longer alive；补跑验证：`nnUNetv2_train 501 3d_fullres 0 --val`，在windows下多进程机制在结束进程后，内存回收和句柄释放有延迟，导致内存(RAM)或进程句柄爆了；直接用模型进行预测（推理）：`nnUNetv2_predict -i <你的输入文件夹> -o <你的输出文件夹> -d 501 -c 3d_fullres -f 0`；不仅要预测结果，还要看验证病例的分数：`nnUNetv2_evaluate_folder -gt "F:\CBCT\nnUNet\nnunetv2\nnUNet_raw\Dataset501_AirwayLandmarks\labelsTr" -pred "F:\CBCT\inference_output" -djfile "F:\CBCT\nnUNet\nnunetv2\nnUNet_preprocessed\Dataset501_AirwayLandmarks\dataset.json"`



缺少hiddenlayer库，尝试安装一下