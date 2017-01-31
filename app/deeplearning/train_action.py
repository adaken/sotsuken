# coding: utf-8

import numpy as np
import chainer
from chainer import Chain, training, optimizers, iterators, serializers, Variable
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions, extension
from chainer.datasets.tuple_dataset import TupleDataset
import matplotlib.pyplot as plt
import json
from app.util.inputmaker import make_input
from collections import namedtuple
import random
from chainer import reporter

def train_MLP():

    class MLP(Chain):
        """多層パーセプトロン"""

        def __init__(self, n_in, n_units, n_out):
            super(MLP, self).__init__(

                # 層の定義
                # Links
                l1=L.Linear(n_in, n_units),    # 入力層 -> 隠れ層 全結合
                l2=L.Linear(n_units, n_units), # 隠れ層 -> 出力層 全結合
                l3=L.Linear(n_units, n_out),   # 出力層 -> 出力   全結合
                )

        def __call__(self, x):

            # xはVariableオブジェクト
            # x.dataは入力ベクトルを要素とした2D配列
            # 長さはバッチの数

            # 順伝播の定義
            # RELUを活性化関数として使う
            # ドロップアウトもここで指定
            h1 = F.dropout(F.relu(self.l1(x)), ratio=.7)  # 入力層の出力
            h2 = F.dropout(F.relu(self.l2(h1)), ratio=.6) # 隠れ層の出力
            y = self.l3(h2)                     # 出力層の出力
            return y

    class ConfusionMatrix(object):
        """コンフュージョン・マトリクス"""

        def __init__(self, n):
            self.sum_matrix = np.zeros([n]*2, dtype=np.int32)

        def __call__(self, conf_matrix):
            self.matrix = conf_matrix
            self.sum_matrix += self.matrix
            self.tp = np.diag(self.matrix) # TP(対角項)

        @property
        def precision(self):
            """適合率"""
            total = self.matrix.sum(axis=1)
            prec = self.tp / total.astype(np.float32)
            prec[np.isnan(prec)] = 1
            return prec

        @property
        def recall(self):
            """再現率"""
            total = self.matrix.sum(axis=0)
            rec = self.tp / total.astype(np.float32)
            rec[np.isnan(rec)] = 1
            return rec

        @property
        def fmeasure(self):
            """F値"""
            prec, rec = self.precision, self.recall
            fm = (2 * rec * prec) / (rec + prec)
            fm[np.isnan(fm)] = 1
            return fm

    class Classifier(chainer.link.Chain):
        """評価関数を定義"""

        from chainer.functions.evaluation import accuracy
        from chainer.functions.loss import softmax_cross_entropy
        np.seterr(divide='ignore', invalid='ignore')

        def __init__(self, predictor,
                     conf_matrix=None,
                     lossfun=softmax_cross_entropy.softmax_cross_entropy,
                     accfun=accuracy.accuracy):
            super(Classifier, self).__init__(predictor=predictor)
            self.lossfun = lossfun
            self.accfun = accfun
            self.conf_matrix = conf_matrix

        def init_var(self):
            self.y = None
            self.loss = None
            self.accuracy = None

        def __call__(self, *args):
            x = args[:-1]  # 入力ベクトル列(Variable)のミニバッチ
            t = args[-1]  # 教師ラベルの配列(Variable)

            self.init_var() # 変数初期化

            # ネットワークの出力の2D配列(Variable)(出力層ノード数 * batchsize)
            self.y = self.predictor(*x)

            self.loss = self.lossfun(self.y, t) # 損失
            self.accuracy = self.accfun(self.y, t) # 正確性
            reporter.report({'loss': self.loss, 'accuracy': self.accuracy}, self)

            # コンフュージョン・マトリクスを更新
            cm = self.get_confusion_matrix(self.y, t)
            self.conf_matrix(cm)

            return self.loss

        def get_confusion_matrix(self, y, t):
            shape = [y.data.shape[1]] * 2
            pred = y.data.argmax(axis=1)
            t = t.data
            cm = np.zeros(shape, dtype=np.int32)
            for i in np.c_[pred, t]:
                cm[tuple(i)] += 1
            return cm

    Xl = namedtuple('Xl', 'path, sheets, col, min_row, sampling, label')
    xls = [Xl(r'E:\work\data\run_1122.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'], 'F', 2,'std', 0),
           Xl(r'E:\work\data\acc_stop_1206.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'], 'F', 2, 'std', 1),
           Xl(r'E:\work\data\jump_128p_174data_fixed.xlsx', ['Sheet'], 'A', 2, 'std', 2),
           #Xl(r'E:\work\data\bicycle_acc_hirano.xlsx', ['Sheet4'], 'F', 2, 3),
           Xl(r'E:\work\data\walk_acc_tateno.xlsx', ['Sheet4'], 'F', 2, 'rand',  3)
           ]
    N = 128
    sample_cnt = 174
    in_vecs = []
    labels = []
    batch_size = 50
    epoch = 1000
    n_in = N / 2
    n_units = 20
    #n_units_2 = 500
    n_out = len(xls)

    # 入力ベクトルを作成
    for xl in xls:
        vecs, label = make_input(xlsx=xl.path, sheetnames=xl.sheets, col=xl.col,
                          min_row=xl.min_row, fft_N=N,sample_cnt=sample_cnt,
                          sampling=xl.sampling, normalizing='std', label=xl.label, log=True)
        map(in_vecs.append, vecs)
        labels += label

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
    print "vector_type   :", type(train_data[0][0])
    print "label_type    :", type(train_data[0][1])

    # 訓練データのイテレータ
    # ループ毎にシャッフルされる
    train_iter = iterators.SerialIterator(train_data, batch_size=batch_size)

    # テストデータのイテレータ
    test_iter = iterators.SerialIterator(test_data, batch_size=batch_size,
                                         repeat=False, shuffle=False)

    # モデル作成
    #model = L.Classifier(MLP(n_in, n_units, n_out))
    conf_matrix = ConfusionMatrix(n_out) # Confusion Matrix
    model = Classifier(MLP(n_in, n_units, n_out), conf_matrix)

    # 最適化の設定
    optimizer = optimizers.SGD() # 確率的勾配降下法
    #optimizer = optimizers.Adam()
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
    trainer.extend(extensions.LogReport(keys=None, log_name=log_name))
    trainer.extend(extensions.ProgressBar())

    trainer.run() # 訓練開始

    print "\n"
    print "Result"
    print "--------------------"
    print "conf_matrix:\n", conf_matrix.matrix
    print "precision:", conf_matrix.precision
    print "recall   :", conf_matrix.recall
    print "f-measure:", conf_matrix.fmeasure

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
