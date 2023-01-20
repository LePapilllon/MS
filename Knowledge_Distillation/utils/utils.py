import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class, targetclass_value):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc_overall = np.diag(hist).sum() / hist.sum() # diag只拿对角线元素（混淆矩阵对角线）

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_user = np.diag(hist) / hist.sum(axis=1)
        acc_producer = np.diag(hist) / hist.sum(axis=0)
    acc_user = np.nanmean(acc_user)
    acc_producer = np.nanmean(acc_producer)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.diag(hist)[targetclass_value] / (hist.sum(axis=0)[targetclass_value] + 1e-8)
        recall = np.diag(hist)[targetclass_value] / (hist.sum(axis=1)[targetclass_value] + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mIoU = np.nanmean(iu)
    # freq = hist.sum(axis=1) / hist.sum()
    return acc_overall, acc_user, acc_producer, mIoU, precision, recall, f1


if __name__ == '__main__':
    print("==========================================================")
    print("示例计算总体精度、用户精度、制图者精度、平均交并比、精确率、召回率、F1")
    label_trues = np.array([[1, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1]])
    label_preds = np.array([[1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 0, 1]])
    hist = _fast_hist(label_trues.flatten(),label_preds.flatten(), 2)
    acc_overall, acc_user, acc_producer, mIoU, precision, recall, f1 = label_accuracy_score\
        (label_trues.flatten(), label_preds.flatten(), 2, 1)
    print("混淆矩阵：")
    print(hist)
    print("OA精度：")
    print(acc_overall * 100)
    print("UA精度：")
    print(acc_user*100)
    print("PA精度：")
    print(acc_producer*100)
    print("MIoU精度：")
    print(mIoU * 100)
    print("precision精度：")
    print(precision * 100)
    print("recall精度：")
    print(recall * 100)
    print("f1精度：")
    print(f1 * 100)
    print("==========================================================")

