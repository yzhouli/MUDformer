import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, average_precision_score, auc, roc_curve


class DataUtil(object):

    @staticmethod
    def max_index(pred_li):
        index = 0
        temp = -1000
        for i, num in enumerate(pred_li):
            if num > temp:
                temp = num
                index = i
        return index

    @staticmethod
    def acc(label_matrix, out_matrix):
        out_matrix = out_matrix.numpy()
        true_total = 0
        acc_li = []
        for i in range(len(label_matrix)):
            if label_matrix[i] == -1:
                continue
            pred = tf.nn.softmax(out_matrix[i])
            pred_index = DataUtil.max_index(pred_li=pred)
            label = 0
            if pred_index == label_matrix[i]:
                true_total += 1
                label = 1
            acc_li.append(label)
        return true_total, acc_li

    @staticmethod
    def normal(predict_li, depth=3):
        result_li = []
        for att_li in predict_li.numpy():
            index = DataUtil.max_index(att_li)
            result_li.append(index)
        result_li = np.asarray(result_li, dtype=np.int32)
        result_li = tf.cast(result_li, dtype=tf.int32)
        result_li = tf.one_hot(result_li, depth=depth)
        return result_li

    @staticmethod
    def AUC(predict_li, label_li):
        p, l = [], []
        for index in range(len(predict_li)):
            p.append(predict_li[index][0])
            p.append(predict_li[index][1])
            l.append(label_li[index][0])
            l.append(label_li[index][1])
        pr_auc = average_precision_score(l, p)
        fpr, tpr, thresholds = roc_curve(l, p)
        roc_auc = auc(fpr, tpr)
        return pr_auc, roc_auc


    @staticmethod
    def evaluation(y_test, y_predict):
        y_predict = DataUtil.normal(predict_li=y_predict,depth=2)
        metrics = classification_report(y_test, y_predict, output_dict=True)
        precision = metrics['0']['precision'], metrics['1']['precision']
        recall = metrics['0']['recall'], metrics['1']['recall']
        f1_score = metrics['0']['f1-score'], metrics['1']['f1-score']
        return precision, recall, f1_score