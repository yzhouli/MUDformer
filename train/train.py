import os

import numpy as np
from tqdm import tqdm


import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, models

from config import Config
from dataset.dataset import Dataset
from util.data_util import DataUtil
from util.util import Util


class Train(object):

    def __init__(self, model, name=None, reverse=False):
        self.model = model
        self.name = name
        self.reverse=reverse
        self.train_db, self.test_db = None, None

    def data_init(self):
        self.train_db = Dataset(mode='train')
        self.test_db = Dataset(mode='test')

    def iteration(self, name, epoch):
        data_train = self.train_db
        if 'test' == name:
            data_train = self.test_db
        optimizer = optimizers.Adam(Config.learning_rate)
        loss_num, acc = 0, 0
        pred_matrix, label_li = None, []
        pbar = tqdm(total=data_train.len() // Config.batch_size + 1)
        user_li = []
        with tf.device('/gpu:0'):
            for item in data_train.get_all():
                text_mat, image_mat, topic_label, mask_mat = data_train.get_item(index_li=item)
                user_li = data_train.get_users(index_li=item, user_li=user_li)
                with tf.GradientTape() as tape:
                    label_matrix = tf.one_hot(topic_label, depth=2)
                    if self.reverse:
                        image_mat, text_mat = text_mat, image_mat
                    out_matrix, MVAE_data = self.model((image_mat, text_mat, mask_mat))
                    # out_matrix = self.model((image_mat, text_mat, mask_mat))
                    # self.model.summary()
                    if pred_matrix is None:
                        pred_matrix = out_matrix
                    else:
                        pred_matrix = tf.concat(values=[pred_matrix, out_matrix], axis=0)
                    [label_li.append(i) for i in topic_label]
                    accuracy, temp = DataUtil.acc(label_matrix=topic_label, out_matrix=out_matrix)
                    acc += accuracy
                    MVAE_loss, MVAE_loss1, class_loss = self.model.compute_loss(out_matrix, label_matrix, image_mat, text_mat, MVAE_data)
                    # self.model.summary()
                    if MVAE_loss is None:
                        loss = class_loss
                    else:
                        loss = Config.MVAE_decay * MVAE_loss + Config.MVAE_decay1 * MVAE_loss1 + Config.class_decay * class_loss
                    # loss = self.model.compute_loss(out_matrix, label_matrix)
                    loss_num += float(loss) * Config.batch_size
                    pbar.desc = f'epoch: {epoch}, name: {name}, loss: {round(loss_num / len(label_li), 4)}, accuracy: {round(acc / len(label_li), 4)}'
                    pbar.update(1)
                    if 'train' == name:
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        label_matrix = tf.one_hot(label_li, depth=2)
        del tape
        data_dict = {'Acc':0}
        accuracy, acc_label_li = DataUtil.acc(label_matrix=label_li, out_matrix=pred_matrix)
        err_li = []
        #print(len(user_li), len(acc_label_li))
        for i in range(len(acc_label_li)):
            if acc_label_li[i] == 0:
                continue
            err_li.append(user_li[i])
        precision, recall, f1_score = DataUtil.evaluation(y_test=label_matrix, y_predict=pred_matrix)
        acc, loss = accuracy / data_train.len(), loss_num / data_train.len()
        print(f'accuracy: {round(acc, 4)}, loss: {round(loss, 4)}')
        print(
            f'non-precision: {round(precision[0], 4)}, non-recall: {round(recall[0], 4)}, non-f1_score: {round(f1_score[0], 4)}')
        print(
            f'spammer-precision: {round(precision[1], 4)}, spammer-recall: {round(recall[1], 4)}, spammer-f1_score: {round(f1_score[1], 4)}')
        pr_auc, roc_auc = DataUtil.AUC(predict_li=pred_matrix, label_li=label_matrix)
        print(f'PR-AUC: {round(pr_auc, 4)}, ROC-AUC: {round(roc_auc, 4)}')
        data_dict = {'Acc': round(acc, 3),
                     'Loss': round(loss, 3),
                     'PR-AUC': round(pr_auc, 4),
                     'ROC-AUC': round(roc_auc, 3),
                     'Non-prec': round(precision[0], 3),
                     'Non-rec': round(recall[0], 3),
                     'Non-f1': round(f1_score[0], 3),
                     'Spammer-prec': round(precision[1], 3),
                     'Spammer-rec': round(recall[1], 3),
                     'Spammer-f1': round(f1_score[1], 3),}
        return data_dict, err_li

    def train(self):
        self.data_init()
        acc_max = -1
        train_li, test_li = [], []
        if self.name is None:
            save_path = f'{Config.save_path}\\{self.model.name}'
        else:
            save_path = f'{Config.save_path}\\{self.name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for epoch in range(Config.epochs):
            data_dict, err_li = self.iteration(name='train', epoch=epoch)
            train_li.append(data_dict)
            if data_dict['Acc'] >= 0.8:
                err_num = Util.save_err(err_li=err_li)
                print(f'error: {err_num}')
                if err_num >= 400:
                    exit(err_num)
            data_dict, err_li = self.iteration(name='test', epoch=epoch)
            test_li.append(data_dict)
            if data_dict['Acc'] >= 0.78:
                err_num = Util.save_err(err_li=err_li)
                print(f'error: {err_num}')
                if err_num >= 400:
                    exit(err_num)
            if data_dict['Acc'] > acc_max:
                acc_max = data_dict['Acc']
                # Util.save_model(model=self.model)
        print(f'max acc: {acc_max}')
        Util.save_data(train_li=train_li, test_li=test_li, save_path=save_path)
