import torch
import torch.nn.functional as F
from utils.utils import *

def teacher_test(model, device, test_loader, num_class, targetclass_value):
    # 不启用 BatchNormalization 和 Dropout
    mean_acc_overall = 0.0
    mean_acc_user = 0.0
    mean_acc_producer = 0.0
    mean_mIoU = 0.0
    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            model.eval()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs.data, 1)

            label_pred = predicted.data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            acc_overall, acc_user, acc_producer, mean_iu, precision, recall, f1 = label_accuracy_score(label_true, label_pred, num_class, targetclass_value)

            mean_acc_overall += acc_overall
            mean_acc_user += acc_user
            mean_acc_producer += acc_producer
            mean_mIoU += mean_iu
            mean_precision += precision
            mean_recall += recall
            mean_f1 += f1

            test_loss += F.cross_entropy(outputs, labels).item()

    test_loss = test_loss / (batch_idx + 1)

    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print('Teacher Test:Loss: %.3f  Acc_overall: %.3f%%  Acc_user: %.3f%%  Acc_producer: %.3f%%  MIoU: %.3f%%  Precision: %.3f%%  Recall: %.3f%%  F1-Score: %.3f%%'
          % (test_loss, (100. * mean_acc_overall / len(test_loader)), (100. * mean_acc_user / len(test_loader)), (100. * mean_acc_producer / len(test_loader)), (100. * mean_mIoU / len(test_loader)), (100. * mean_precision / len(test_loader)), (100. * mean_recall / len(test_loader)), (100. * mean_f1 / len(test_loader))))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    test_result = np.array(
        [test_loss, (100. * mean_acc_overall / len(test_loader)), (100. * mean_acc_user / len(test_loader)),
         (100. * mean_acc_producer / len(test_loader)), (100. * mean_mIoU / len(test_loader)),
         (100. * mean_precision / len(test_loader)), (100. * mean_recall / len(test_loader)),
         (100. * mean_f1 / len(test_loader))])

    return test_result