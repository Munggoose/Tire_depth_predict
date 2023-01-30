from sklearn.metrics import f1_score


def F1_score(pred,true):
    return f1_score(true,pred,average='micro')