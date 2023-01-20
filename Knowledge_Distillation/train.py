import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from torchvision import models
import matplotlib.pyplot as plt

from voc_seg_data import VOC_SEG_Dataset

from model.student_model import Student
from model.student_model import Student

from train.teacher_train import teacher_train
from train.student_train import student_train
from train.teacherstudent_KD_train import teacherstudent_KD_train

from val.teacher_val import teacher_val
from val.student_val import student_val
from val.teacherstudent_KD_val import teacherstudent_KD_val

# 这里取消警告
import warnings
warnings.filterwarnings("ignore")

def teacher_main(epoches, root, batch_size, width, height, num_class, targetclass_value, lr, momentum, gpus):
    print("==========================================================================================================")
    print("Teacher model training and valing:")
    voc_train = VOC_SEG_Dataset(root, width, height, train=True)
    voc_val = VOC_SEG_Dataset(root, width, height, train=False)
    train_loader = DataLoader(voc_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(voc_val, batch_size=batch_size, shuffle=True)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_history = []
    # 实例化模型
    model = Student(n_channels=3, num_classes=num_class).to(device)

    # 使用GPU才能用
    if gpus is not None:
        model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    # 选取优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epoches + 1):
        teacher_train(model, device, train_loader, optimizer, epoch, num_class)
        val_result = teacher_val(model, device, val_loader, num_class, targetclass_value)

        teacher_history.append(val_result)

    # 保存模型,state_dict:Returns a dictionary containing a whole state of the module.
    torch.save(model.state_dict(), r'model_parameters/teacher.pt')

    return model, teacher_history


def student_main(epoches, root, batchsize, width, height, num_class, targetclass_value, lr, momentum, gpus):
    print("==========================================================================================================")
    print("Student model training and testing:" )
    voc_train = VOC_SEG_Dataset(root, width, height, train=True)
    voc_val = VOC_SEG_Dataset(root, width, height, train=False)
    train_loader = DataLoader(voc_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(voc_val, batch_size=batch_size, shuffle=True)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_history = []
    # 实例化模型
    model = Student(n_channels=3, num_classes=num_class).to(device)

    # 使用GPU才能用
    if gpus is not None:
        model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    optimizer= torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epoches + 1):
        student_train(model, device, train_loader, optimizer, epoch, num_class)
        val_result = student_val(model, device, val_loader, num_class, targetclass_value)

        student_history.append(val_result)

    # 保存模型,state_dict:Returns a dictionary containing a whole state of the module.
    torch.save(model.state_dict(), r'model_parameters/student.pt')

    return model, student_history


def teacherstudent_KD_main(teacher_model, epoches, temperature, alpha, root, batch_size, width, height, num_class, targetclass_value, lr, momentum, gpus):
    print("==========================================================================================================")
    print("TeacherStudent KD model training and testing:")
    voc_train = VOC_SEG_Dataset(root, width, height, train=True)
    voc_val = VOC_SEG_Dataset(root, width, height, train=False)
    train_loader = DataLoader(voc_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(voc_val, batch_size=batch_size, shuffle=True)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacherstudent_KD_history = []
    # 实例化模型
    model = Student(n_channels=3, num_classes=num_class).to(device)

    # 使用GPU才能用
    if gpus is not None:
        model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    # 选取优化器
    optimizer= torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epoches + 1):
        teacherstudent_KD_train(model, teacher_model, device, train_loader, optimizer, epoch, num_class, temperature, alpha)
        val_result = teacherstudent_KD_val(model, device, val_loader, num_class, targetclass_value)

        teacherstudent_KD_history.append(val_result)

    # 保存模型,state_dict:Returns a dictionary containing a whole state of the module.
    torch.save(model.state_dict(), r'model_parameters/teacherstudent_KD.pt')

    return model, teacherstudent_KD_history


if __name__ == '__main__':
    epoches = 100
    root = r"data"
    batch_size = 8

    # 设置所用的GPU
    # gpus = np.array([0, 1, 2, 3])
    gpus = None

    width = 512
    height = 512
    num_class = 2
    lr = 1e-4
    momentum = 0.9

    temperature = 5
    alpha = 0.5
    targetclass_value = 1

    teacher_model, teacher_history = teacher_main(epoches, root, batch_size, width, height, num_class, targetclass_value, lr, momentum, gpus)
    np.savetxt(r"model_result/unet_val_teacher_result.txt", teacher_history, delimiter=',')

    student_model, student_history = student_main(epoches, root, batch_size, width, height, num_class, targetclass_value, lr, momentum, gpus)
    np.savetxt(r"model_result/vgg_val_student_result.txt", student_history, delimiter=',')

    teacherstudent_KD_model, teacherstudent_KD_history = teacherstudent_KD_main(teacher_model, epoches, temperature, alpha, root, batch_size, width, height, num_class, targetclass_value, lr, momentum,gpus)
    np.savetxt(r"model_result/val_teacherstudent_KD_result.txt", teacherstudent_KD_history, delimiter=',')
