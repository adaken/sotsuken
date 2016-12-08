# coding: utf-8

import numpy as np
from chainer import Chain, training, optimizers, iterators, serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets.tuple_dataset import TupleDataset
import matplotlib.pyplot as plt
import json
from util.util import make_input, timecounter
from collections import namedtuple
import random

@timecounter
def train_MLP():
    """多層パーセプトロンを訓練

    """

    class MLP(Chain):

        def __init__(self, n_in, n_units, n_out):
            super(MLP, self).__init__(

                # 層の定義
                # Links
                l1=L.Linear(n_in, n_units),    # 入力層  -> 隠れ層1
                l2=L.Linear(n_units, n_units), # 隠れ層1 -> 隠れ層2
                l3=L.Linear(n_units, n_out),   # 隠れ層2 -> 出力層
                )

        def __call__(self, x):

            # 順伝播の定義
            # RELUを活性化関数として使う
            # ドロップアウトもここで指定
            h1 = F.dropout(F.relu(self.l1(x)))  # 入力層の出力
            h2 = F.dropout(F.relu(self.l2(h1))) # 隠れ層の出力
            y = self.l3(h2)                     # 出力層の出力
            return y


    Xl = namedtuple('Xl', 'path, sheets, col, min_row, label')
    xls = [Xl(r'E:\work\data\run_1122.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'], 'F', 2, 0),
           Xl(r'E:\work\data\acc_stop_1206.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'], 'F', 2, 1),
           Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', ['Sheet'], 'A', 2, 2)]
    N = 128
    sample_cnt = 174
    in_vecs = []
    labels = []
    batch_size = 50
    epoch = 1000
    n_in = N / 2
    n_units = 500
    n_out = len(xls)

    for xl in xls:
        vecs = make_input(xlsx=xl.path, sheetnames=xl.sheets, col=xl.col,
                          min_row=xl.min_row, fft_N=N,sample_cnt=sample_cnt, log=True)
        map(in_vecs.append, vecs)
        labels += [xl.label] * len(vecs)

    print "vecs_len  :", len(in_vecs)
    print "vec_len   :", len(in_vecs[0])
    print "labels_len:", len(labels)

    # 入力ベクトルをシャッフル
    tmp = np.c_[in_vecs, labels] # ラベルと結合
    random.shuffle(tmp)          # シャッフル
    in_vecs = tmp[:, :-1]        # データ - 2D
    labels  = tmp[:, -1]         # ラベル - 1D

    # 型を変換
    in_vecs = np.array(in_vecs, dtype=np.float32)
    labels  = np.array(labels, dtype=np.int32)

    n = 50
    train_data   = in_vecs[:-n] # 後ろからnデータ前まで訓練用
    test_data    = in_vecs[-n:] # 後ろからnデータをテスト用
    train_labels = labels[:-n]
    test_labels  = labels[-n:]
    print "train_data_len: {}".format(len(train_data))
    print "test_data_len : {}".format(len(test_data))

    # 型を変換
    train_data = TupleDataset(train_data, train_labels)
    test_data  = TupleDataset(test_data, test_labels)
    print type(train_data[0][0]), type(train_data[0][1])

    # 訓練データのイテレータ
    # ループ毎にシャッフルされる
    train_iter = iterators.SerialIterator(train_data, batch_size=batch_size)

    # テストデータのイテレータ
    test_iter = iterators.SerialIterator(test_data, batch_size=batch_size,
                                         repeat=False, shuffle=False)

    # モデル作成
    model = L.Classifier(MLP(n_in, n_units, n_out)
                         #lossfan=softmax_cross_entropy.softmax_cross_entropy,
                         #accfun=accuracy.accuracy
                         )

    # 最適化の設定
    optimizer = optimizers.SGD() # 確率的勾配降下法
    optimizer.setup(model)       # 最適化の準備

    updater = training.StandardUpdater(train_iter, optimizer)

    # モデルを訓練
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    # ログファイル
    log_name = r'E:\chainer\train_action_log.json'

    # 拡張
    # 各エポックの終わりに呼び出される
    # テストデータに対するモデルをループごとに評価
    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport(log_name=log_name))
    #trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run() # 訓練開始

    # ログをグラフ化
    def as_list(d):
        return (d['epoch'], d['validation/main/loss'], d['validation/main/accuracy'],
                d['main/loss'], d['main/accuracy'])

    # jsonからリストに
    log = json.load(file(log_name), object_hook=as_list)                # ログを読み込む
    epoch, test_loss, test_acc, train_loss, train_acc = np.array(log).T # リストを転置

    #plt.style.use('ggplot')
    plt.hold(True)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylim(0, 2)
    plt.plot(epoch, test_loss)
    plt.plot(epoch, train_loss)
    plt.plot(epoch, test_acc)
    plt.plot(epoch, train_acc)
    plt.legend(['test_loss', 'train_loss', 'test_accuracy', 'train_accuracy'],
               loc='best')
    plt.title("MLP, Use Dropout")
    plt.plot()
    plt.show()

    # モデルを保存
    #serializers.save_npz(r'E:\chainer\my.model', model)

if __name__ == '__main__':
    train_MLP()