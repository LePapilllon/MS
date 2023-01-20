import math
import torch.nn.functional as F
import torch
from utils.utils import *

def teacher_train(model, device, train_loader, optimizer, epoch, num_class):
    # 启用 BatchNormalization 和 Dropout
    print('\nTeacher Epoch: %d' % (epoch))
    model.train()
    step_schedule= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                              verbose=False, threshold=0.0001, threshold_mode='rel',
                                                              cooldown=0, min_lr=0, eps=1e-08)
    trained_samples = 0
    sum_loss = 0.0
    aux_layers = [6, 12, 18, 24] # SETR
    for batch_idx, (images, labels) in enumerate(train_loader):
        length = len(train_loader)
        # 搬到指定gpu或者cpu设备上运算
        images, labels = images.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)  # torch.size([batch_size, num_class, width, height])
        # (outputs, aux_layers_outputs) = model(images, aux_layers)  # torch.size([batch_size, num_class, width, height]) # SETR
        # 计算损失
        # loss = F.cross_entropy(F.softmax(outputs, dim=1), labels.long())
        loss = F.cross_entropy(outputs, labels.long())
        # 误差反向传播
        loss.backward()
        # 梯度更新一步
        optimizer.step()

        sum_loss += loss.item()

        # 学习率调整
        step_schedule.step(sum_loss)

        # 统计已经训练的数据量
        trained_samples += len(images)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print('\rTeacher Train epoch: {} {}/{} [{}]{}%'.format(epoch, trained_samples, len(train_loader.dataset),
                                                       '-' * progress + '>', progress * 2), end='')
    print('\nTeacher [epoch:%d, iter:%d] Train_Loss: %.04f' % (epoch, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1)))