#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : szu-hwj

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
import os
import tarfile
import numpy as np

# folder = r'E:\2018上\自学方向\磐创\自己写的文章\使用text-cnn处理情感分析'
# os.chdir(folder)



# 有些数据是含有html标签的，需要去除
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(filetype):
    """
    filetype: 'train' or 'test'
    return:
    all_texts: filetype数据集文本
    all_labels: filetype数据集标签
    """
    # 标签1表示正面，0表示负面
    all_labels = [1]*12500 + [0]*12500
    all_texts = []
    file_list = []
    path = r'./aclImdb/'
    # 读取正面文本名
    pos_path = path + filetype + '/pos/'
    for file in os.listdir(pos_path):
        file_list.append(pos_path+file)
    # 读取负面文本名
    neg_path = path + filetype + '/neg/'
    for file in os.listdir(neg_path):
        file_list.append(neg_path+file)
    # 将所有文本内容加到all_texts
    for file_name in file_list:
        with open(file_name, encoding='utf-8') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    return all_texts, all_labels


def preprocessing(train_texts, train_labels, test_texts, test_labels):
    tokenizer = Tokenizer(num_words=2000)  # 建立一个2000个单词的字典
    tokenizer.fit_on_texts(train_texts)
    # 对每一句影评文字转换为数字列表，使用每个词的编号进行编号
    x_train_seq = tokenizer.texts_to_sequences(train_texts)
    x_test_seq = tokenizer.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(x_train_seq, maxlen=150)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=150)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    return x_train, y_train, x_test, y_test


def text_cnn(maxlen=150, max_features=2000, embed_size=32):
    # Inputs
    comment_seq = Input(shape=[maxlen], name='x_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=1, activation='sigmoid')(output)

    model = Model([comment_seq], output)
    #     adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

if __name__ == '__main__':
    if not os.path.exists('./aclImdb'):
        tfile = tarfile.open(r'./aclImdb_v1.tar.gz', 'r:gz')  # r;gz是读取gzip压缩文件
        result = tfile.extractall('./')  # 解压缩文件到当前目录中
    train_texts, train_labels = read_files('train')
    test_texts, test_labels = read_files('test')
    x_train, y_train, x_test, y_test = preprocessing(train_texts, train_labels, test_texts, test_labels)
    model = text_cnn()
    batch_size = 128
    epochs = 20
    model.fit(x_train, y_train,
              validation_split=0.1,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

