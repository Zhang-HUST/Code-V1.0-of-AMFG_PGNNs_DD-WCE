import torch


def get_accuracy_torch(predict_labels, true_labels, decimal=3):
    predict_labels = torch.cat(predict_labels, dim=0).view(-1)
    true_labels = torch.cat(true_labels, dim=0).view(-1)
    acc = torch.sum(predict_labels == true_labels) / true_labels.shape[0]
    acc = acc.item()

    return round(acc * 100.0, decimal)
