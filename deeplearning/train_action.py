# coding: utf-8

import numpy as np
import chainer
from chainer import Chain, training, optimizers, iterators
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import matplotlib.pyplot as plt
import json
from util import fft
from util.excelwrapper import ExcelWrapper
from collections import namedtuple

def train_MLP():
    """多層パーセプトロンを訓練

    """

    class MLP(Chain):
        def __init__(self, n_in, n_units, n_out):
            super(MLP, self).__init__(

                # Links
                l1=L.Linear(n_in, n_units),    # 入力層
                l2=L.Linear(n_units, n_units), # 隠れ層
                l3=L.Linear(n_units, n_out),   # 出力層
                )

        def __call__(self, x):

            # RELUを活性化関数として使う
            # ドロップアウトもここで指定
            h1 = F.dropout(F.relu(self.l1(x)))  # 入力層の出力
            h2 = F.dropout(F.relu(self.l2(h1))) # 隠れ層の出力
            y = self.l3(h2)                     # 出力層の出力
            return y


    Xl = namedtuple('Xl', 'path, sheets, col, row_range, label')
    xls = [Xl(r'E:\work\data\new_run.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'], 'F', (2, None), 1),
           Xl(r'E:\work\data\walk.xlsx', ['Sheet4'], 'F', (2, None), 2),
           Xl(r'E:\work\data\brisk_walk.xlsx', ['Sheet4', 'Sheet5', 'Sheet6'], 'F', (2, None), 3)]
    N = 128
    vecs = []

    for xl in xls:
        wb = ExcelWrapper(xl.path)
        for sheet in xl.sheets:
            ws = wb.get_sheet(sheet)
            acces = (ws.iter_part_col(col=xl.col, length=N, row_range=xl.row_range, log=True))
            fftdata = fft.fftn(arrs=list(acces), fft_N=N, wf='hanning')
            vecs.append((vec, xl.label) for vec in fftdata)

    print "vecs: {}".format(len(vecs))
    # [tuple(array(), label)...]
    train_data = vecs[:-100] # 後ろから100データ前まで訓練用
    test_data =  vecs[-100:] # 後ろから100データをテスト用
    print "train_data: {}".format(len(train_data))
    print "test_data : {}".format(len(test_data))

    # 訓練データのイテレータ
    # ループ毎にシャッフルされる
    train_iter = iterators.SerialIterator(train_data, batch_size=100)

    # テストデータのイテレータ
    test_iter = iterators.SerialIterator(test_data, batch_size=100, repeat=False, shuffle=False)

    model = L.Classifier(predictor=MLP(n_in=N, n_units=1000, n_out=3),
                         #lossfan=softmax_cross_entropy.softmax_cross_entropy,
                         #accfun=accuracy.accuracy
                         )

    # 最適化の設定
    optimizer = optimizers.SGD() # 確率的勾配降下法
    optimizer.setup(model)       # 最適化の準備

    updater = training.StandardUpdater(train_iter, optimizer)

    # モデルを訓練
    # 訓練を20エポック
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')

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

if __name__ == '__main__':
    train_MLP()