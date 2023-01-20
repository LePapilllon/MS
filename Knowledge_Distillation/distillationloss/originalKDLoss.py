import torch.nn as nn
import torch.nn.functional as F

def OriginalKDLoss(student_output, label, teacher_output, temperature, alpha):
    """
        student_output: 学生网络预测的概率分布
        label: 实际标签
        teacher_output: 老师网络预测网络的概率分布
        temperature: 蒸馏温度
        param alpha: 损失调整因子
        return: KD_Loss
    """
    # 源代码使用的是重新定义的KD_Loss 引入了KL散度
    # 来自文章Exploring Knowledge Distillation of Deep Neural Networks for Efficient Hardware Solutions
    kl_student_teacher = nn.KLDivLoss()(F.log_softmax(student_output / temperature, dim=1),
                                        F.softmax(teacher_output / temperature, dim=1)) * (temperature * temperature) * alpha
    student_loss = F.cross_entropy(student_output, label) * (1 - alpha)
    KD_Loss = kl_student_teacher + student_loss

    # 最原始的KD_Loss来源于Hinton文章 Distilling the Knowledge in a Neural Network 但感觉从结果上来说不太对
    # distillation_loss = F.cross_entropy(F.softmax(student_output / temperature, dim=1),
    #                                     F.softmax(teacher_output / temperature, dim=1)) * alpha
    # student_loss = F.cross_entropy(F.softmax(student_output, dim=1), label.long()) * (1-alpha)
    # KD_Loss = distillation_loss + student_loss

    # 还有KD_Loss来源于文章 Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy
    # 三个交叉熵损失相加：教师网络Softmax输出的交叉熵loss、学生网络Softmax输出的交叉熵loss、教师网络数值输出与学生网络Softmax输出的交叉熵loss
    # distillation_loss = F.cross_entropy(F.softmax(student_output / temperature, dim=1),
    #                                     F.softmax(teacher_output / temperature, dim=1)) * alpha
    # student_loss = F.cross_entropy(F.softmax(student_output), label)
    # teacher_loss = F.cross_entropy(teacher_output, label)
    # KD_Loss = distillation_loss + student_loss + teacher_loss

    return KD_Loss
