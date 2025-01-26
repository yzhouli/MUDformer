import json
import tensorflow as tf
import numpy as np

from config import Config
from util.util import Util


class Dataset(object):

    def __init__(self, mode):
        self.path_ref = Config.path_ref
        if 'train' == mode:
            self.data = json.load(open(f'{Config.path_ref}/train.json'))
        else:
            self.data = json.load(open(f'{Config.path_ref}/test.json'))

    def process_index(self, index):
        index = tf.cast(index, tf.int32)
        return index

    def get_all(self):
        index_li = np.asarray([int(i) for i in self.data])
        data_db = tf.data.Dataset.from_tensor_slices(index_li)
        data_db = data_db.map(self.process_index).shuffle(10000).batch(Config.batch_size)
        return data_db

    def get_item(self, index_li):
        index_li = index_li.numpy()
        text_mat, image_mat, mask_mat, topic_label = [], [], [], []
        for index in index_li:
            text_li, image_li, label, mask_li = self.iteration(index=index)
            text_mat.append(text_li)
            image_mat.append(image_li)
            topic_label.append(label)
            mask_mat.append(mask_li)
        text_mat = np.asarray(text_mat, dtype=np.float32)
        image_mat = np.asarray(image_mat, dtype=np.int32)
        topic_label = np.asarray(topic_label, dtype=np.int32)
        mask_mat = np.asarray(mask_mat, dtype=np.float32)
        return text_mat, image_mat, topic_label, mask_mat

    def iteration(self, index):
        index = str(index)
        user_path = self.path_ref + self.data[index]['path']
        text_li, image_li, mask_li = Util.load_embedding(user_path)
        label = self.data[index]['label']
        return text_li, image_li, label, mask_li

    def len(self):
        return len(self.data.keys())

    def get_users(self, index_li, user_li=None):
        if user_li is None:
            user_li = []
        for index in index_li:
            index = str(int(index))
            user_path = self.path_ref + self.data[index]['path']
            user_li.append(user_path)
        return user_li
