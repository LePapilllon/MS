import torch
import torch.nn.functional as F
from utils.utils import *

def teacherstudent_KD_val(model, device, val_loader, num_class, targetclass_value):
    # 不启用 BatchNormalization 和 Dropout
    mean_acc_overall = 0.0
    mean_acc_user = 0.0
    mean_acc_producer = 0.0
    mean_mIoU = 0.0
    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            model.eval()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)

            label_pred = predicted.data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            acc_overall, acc_user, acc_producer, mIoU, precision, recall, f1 = label_accuracy_score(label_true, label_pred, num_class, targetclass_value)

            mean_acc_overall += acc_overall
            mean_acc_user += acc_user
            mean_acc_producer += acc_producer
            mean_mIoU += mIoU
            mean_precision += precision
            mean_recall += recall
            mean_f1 += f1

            val_loss += F.cross_entropy(outputs, labels).item()

    val_loss = val_loss / (batch_idx + 1)

    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print('TeacherStudent KD Val:Loss: %.3f  Acc_overall: %.3f%%  Acc_user: %.03f%%  Acc_producer: %.3f%%  MIoU: %.3f%%  Precision: %.3f%%  Recall: %.3f%%  F1-Score: %.3f%%'
          % (val_loss, (100. * mean_acc_overall / len(val_loader)), (100. * mean_acc_user / len(val_loader)), (100. * mean_acc_producer / len(val_loader)), (100. * mean_mIoU / len(val_loader)), (100. * mean_precision / len(val_loader)), (100. * mean_recall / len(val_loader)), (100. * mean_f1 / len(val_loader))))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    val_result = np.array(
        [val_loss, (100. * mean_acc_overall / len(val_loader)), (100. * mean_acc_user / len(val_loader)),
         (100. * mean_acc_producer / len(val_loader)), (100. * mean_mIoU / len(val_loader)),
         (100. * mean_precision / len(val_loader)), (100. * mean_recall / len(val_loader)),
         (100. * mean_f1 / len(val_loader))])

    return val_result

