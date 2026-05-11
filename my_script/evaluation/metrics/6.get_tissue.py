import os
import numpy as np
import SimpleITK as sitk
import vtk

# ================= 配置区域 =================
# 1. 随便挑一个你想要可视化的病人的 label.nii.gz 文件路径
LABEL_PATH = "/data1/xyh/data/nnUNet/nnUNet_results/Dataset511_AirwayLandmarks/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/validation_postprocessed_LCC/Coelho_R.S.  _1972_12_15.nii.gz" # 请替换为你的实际文件

# 2. 生成的 3D 模型保存路径 (保存在同级目录下，后缀为 .vtk)
OUTPUT_VTK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Coelho_R.S.  _1972_12_15_Pharyngeal_Area.vtk")

# 3. 目标点的 ID
PNS_ID = 4
TUV_ID = 7
BEP_ID = 3
TEE_ID = 5
# ============================================

def get_landmark_centroid(shape_filter, label_id):
    if shape_filter.HasLabel(label_id):
        # 注意：这里 SimpleITK 获取的坐标顺序，我们直接给到 VTK
        return np.array(shape_filter.GetCentroid(label_id))
    return None

def main():
    if not os.path.exists(LABEL_PATH):
        print(f"❌ 找不到文件: {LABEL_PATH}")
        return

    print("🚀 正在读取标签文件提取 3D 坐标...")
    label_img = sitk.ReadImage(LABEL_PATH)
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(label_img)

    p_pns = get_landmark_centroid(shape_filter, PNS_ID)
    p_tuv = get_landmark_centroid(shape_filter, TUV_ID)
    p_bep = get_landmark_centroid(shape_filter, BEP_ID)
    p_tee = get_landmark_centroid(shape_filter, TEE_ID)

    if any(p is None for p in [p_pns, p_tuv, p_bep, p_tee]):
        print("❌ 图像中缺失必要的 Landmark，无法生成 3D 表面！")
        return

    # 打印算出的坐标，让你心里有数
    print(f"✅ 成功提取坐标:")
    print(f"   PNS: {p_pns}")
    print(f"   TUV: {p_tuv}")
    print(f"   BEP: {p_bep}")
    print(f"   TEE: {p_tee}")

    print("\n🔨 正在将 4 个点构建为双三角形 3D 网格模型...")
    
    # 1. 创建 VTK 的点集
    points = vtk.vtkPoints()
    points.InsertNextPoint(p_pns[0], p_pns[1], p_pns[2]) # Point 0
    points.InsertNextPoint(p_tuv[0], p_tuv[1], p_tuv[2]) # Point 1
    points.InsertNextPoint(p_bep[0], p_bep[1], p_bep[2]) # Point 2
    points.InsertNextPoint(p_tee[0], p_tee[1], p_tee[2]) # Point 3

    # 2. 创建单元集 (Cells)，我们将用两个三角形拼成这个四边形面积
    triangles = vtk.vtkCellArray()

    # 三角形 1: PNS(0) - TUV(1) - BEP(2)
    triangle1 = vtk.vtkTriangle()
    triangle1.GetPointIds().SetId(0, 0)
    triangle1.GetPointIds().SetId(1, 1)
    triangle1.GetPointIds().SetId(2, 2)
    triangles.InsertNextCell(triangle1)

    # 三角形 2: PNS(0) - BEP(2) - TEE(3)
    triangle2 = vtk.vtkTriangle()
    triangle2.GetPointIds().SetId(0, 0)
    triangle2.GetPointIds().SetId(1, 2)
    triangle2.GetPointIds().SetId(2, 3)
    triangles.InsertNextCell(triangle2)

    # 3. 将点和三角形组装成 PolyData (3D 面片数据)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    # 4. 将生成的 3D 模型写入 .vtk 文件
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(OUTPUT_VTK)
    writer.SetInputData(polydata)
    writer.Write()

    print(f"🎉 成功！3D 面积模型已保存至: {OUTPUT_VTK}")
    print("👉 请直接将此 .vtk 文件拖入 3D Slicer 中进行查看！")

if __name__ == "__main__":
    main()