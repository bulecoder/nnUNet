# 操作说明

### 环境配置
```
conda create -n nnunetv2 python=3.10 -y
conda activate nnunetv2
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2

# 配置conda环境变量
conda env config vars set nnUNet_raw=F:\nnUNet\nnUNet_raw
conda env config vars set nnUNet_preprocessed=F:\nnUNet\nnUNet_preprocessed
conda env config vars set nnUNet_results=F:\nnUNet\nnUNet_results
# 配置完成以后重新激活conda环境
# 验证是否成功
echo %nnUNet_raw%
echo %nnUNet_preprocessed%
echo %nnUNet_results%

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
* 开始训练，
