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

from voc_seg_modeltest import VOC_SEG_Dataset
from model.teacher_model import Teacher
from model.student_model import Student

from test.teacher_test import teacher_test
from test.student_test import student_test
from test.teacherstudent_KD_test import teacherstudent_KD_test

# 这里取消警告
import warnings
warnings.filterwarnings("ignore")

def teacher_modeltest(root, batchsize, width, height, num_class, targetclass_value, gpus):
    print("==========================================================================================================")
    print("Teacher model test:")
    voc_test = VOC_SEG_Dataset(root, height, width)
    test_loader = DataLoader(voc_test, batch_size=batchsize, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_history = []
    model = Teacher(n_channels=3, num_classes=num_class).to(device)
    model.load_state_dict(torch.load(r'model_parameters/teacher.pt'))

    if gpus is not None:
        model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    teacher_result = teacher_test(model, device, test_loader, num_class, targetclass_value)

    return teacher_result

def student_modeltest(root, batchsize, width, height, num_class, targetclass_value, gpus):
    print("==========================================================================================================")
    print("Student model test:")
    voc_test = VOC_SEG_Dataset(root, height, width)
    test_loader = DataLoader(voc_test, batch_size=batchsize, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Student(n_channels=3, num_classes=num_class).to(device)
    model.load_state_dict(torch.load(r'model_parameters/student.pt'))

    if gpus is not None:
        model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    student_result = student_test(model, device, test_loader, num_class, targetclass_value)

    return student_result

def teacherstudent_KD_modeltest(root, batchsize, width, height, num_class, targetclass_value, gpus):
    print("==========================================================================================================")
    print("TeacherStudent_KD model test:")
    voc_test = VOC_SEG_Dataset(root, height, width)
    test_loader = DataLoader(voc_test, batch_size=batchsize, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Student(n_channels=3, num_classes=num_class).to(device)
    model.load_state_dict(torch.load(r'model_parameters/teacherstudent_KD.pt'))

    if gpus is not None:
        model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    teacherstudent_KD_result = teacherstudent_KD_test(model, device, test_loader, num_class, targetclass_value)

    return teacherstudent_KD_result

if __name__ == '__main__':
    root = r"data"
    batch_size = 8

    # 设置所用的GPU
    # gpus = np.array([0, 1, 2, 3])
    gpus = None

    width = 512
    height = 512
    num_class = 2
    targetclass_value = 1

    teacher_result = teacher_modeltest(root, batch_size, width, height, num_class, targetclass_value, gpus)

    student_result = student_modeltest(root, batch_size, width, height, num_class, targetclass_value, gpus)

    teacherstudent_KD_result = teacherstudent_KD_modeltest(root, batch_size, width, height, num_class, targetclass_value, gpus)

    np.savetxt(r"model_result/test_teacher_result.txt", teacher_result, delimiter=',')
    np.savetxt(r"model_result/test_student_result.txt", student_result, delimiter=',')
    np.savetxt(r"model_result/test_teacherstudent_KD_result.txt", teacherstudent_KD_result, delimiter=',')

    print(" ")
    print("==========================================================================================================")
    print("All models comparison(F1-Score):")
    print("Teacher: %.3f%% , Student: %.3f%% , TeacherStudent_KD: %.3f%%" % (teacher_result[-1], student_result[-1] , teacherstudent_KD_result[-1]))

