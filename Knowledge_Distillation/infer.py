import sys
import torch
from osgeo import gdal

import torch.utils.data
from model.teacher_model import Teacher
from model.student_model import Student

def infer(image_path, infer_model="teacherstudent_KD"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = gdal.Open(image_path)

    img_width = img.RasterXSize  # 栅格矩阵的列数
    img_height = img.RasterYSize  # 栅格矩阵的行数
    img_data = img.ReadAsArray(0, 0, img_width, img_height)  # 将数据写成数组，对应栅格矩阵
    img_tensor = torch.from_numpy(img_data).float()
    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.shape)

    if infer_model != "teacherstudent_KD" and infer_model != "teacher" and infer_model != "student":
        print("model selection error!")
        sys.exit(1)

    elif infer_model == "teacherstudent_KD":
        model = Student(num_classes=2).to(device)
        model.load_state_dict(torch.load(r'model_parameters/teacherstudent_KD.pt'))

    elif infer_model == "teacher":
        model = Teacher(num_classes=2).to(device)
        model.load_state_dict(torch.load(r'model_parameters/teacher.pt'))

    elif infer_model == "student":
        model = Student(num_classes=2).to(device)
        model.load_state_dict(torch.load(r'model_parameters/student.pt'))

    model.eval()
    infer_predicted = model(img_tensor)
    infer_result = torch.argmax(infer_predicted.data, 1)
    infer_result = infer_result.squeeze(0)
    infer_result = infer_result.cpu().numpy()

    return infer_result

if __name__ == '__main__':
    image_path = r"infer_data/infer_image/myimg_981.tif"
    label_path = r"infer_data/infer_image/mylabel_981.tif"
    # infer_model = "teacher"
    # infer_model = "student"
    infer_model = "teacherstudent_KD"
    infer_result = infer(image_path, infer_model=infer_model)

    # infer_picture = cv2.imwrite(r"infer_data/infer_result/myresult" + image_path[29:-4] + ".tif", infer_result)

    img = gdal.Open(image_path)
    img_proj = img.GetProjection()  # 地图投影信息
    print("projection:" + str(img_proj))
    img_geotrans = img.GetGeoTransform()  # 仿射矩阵
    print("coordinate:" + str(img_geotrans))

    img_width = img.RasterXSize  # 栅格矩阵的列数
    img_height = img.RasterYSize  # 栅格矩阵的行数

    driver = gdal.GetDriverByName("GTiff")
    img_output = driver.Create(r"infer_data/infer_result/myresult_" + infer_model + "_" + image_path[29:-4] + ".tif", img_width, img_height, 1, gdal.GDT_Int32)

    result = img_output.GetRasterBand(1)
    result.WriteArray(infer_result * 1)
    img_output.SetProjection(img.GetProjection())
    img_output.SetGeoTransform(img_geotrans)
    print("output finished in : infer_data/infer_result/myresult_" + infer_model + "_" + image_path[29:-4] + ".tif")
