# coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import matplotlib.pyplot as plt
import json
from util.util import timecounter

def test1():

    # Variableオブジェクトを作成
    x_data = np.array([5, 6, 7], dtype=np.float32)
    x = Variable(x_data)

    z = 2*x**2

    # 関数  :     y = x2 + z
    # 導関数: f'(x) = 2x + 4x
    y = x**2 + z

    print "計算結果: {}".format(y.data)

    y.grad = np.ones(3, dtype=np.float32) # 初期エラー値
    y.backward(retain_grad=True)          # BP

    # たぶん偏微分したもの
    print "勾配x   : {}".format(x.grad) # der(y) / der(x)
    print "勾配z   : {}".format(z.grad) # der(y) / der(z)

def test2():

    # Linkは高階関数
    l = L.Linear(3, 2) # 3D -> 2Dのアフィン変換, f(x) = Wx + b
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    y = l(x)

    print "元データ:\n{}".format(x.data)
    print "アフィン変換:\n{}".format(y.data)

    # パラメータの勾配は蓄積されるので計算を更新する場合は初期化すべし
    l.zerograds()                              # 勾配を初期化
    y.grad = np.ones((2, 2), dtype=np.float32) # 初期エラー値
    y.backward()

    print "勾配W:\n{}".format(l.W.grad) # der(y) / der(W)
    print "勾配b:\n{}".format(l.b.grad) # der(y) / der(b)

def test3():

    # ChainはLinkをまとめたもの
    # また、ChainはLinkを継承している
    class MyChain(Chain):
        def __init__(self):
            super(MyChain, self).__init__(
                l1=L.Linear(4, 3), # 4D->3Dの線形写像(内積)
                l2=L.Linear(3, 2),
            )

        def __call___(self, x):
            h = self.l1(x)
            return self.l2(h)

    # ネットワークを学習(Linkのパラメータ最適化)する方法を定義
    model = MyChain()
    optimizer = optimizers.SGD() # 確率的勾配降下法
    optimizer.setup(model)       # 最適化の準備
    # フック関数として正則化(荷重減衰)を設定 -> ドロップコネクト？
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    model.zerograds()            # 勾配を初期化
    # 勾配の計算処理を手動でここでやる
    # もしくはupdate()に損失関数を渡す
    optimizer.update()           # 勾配を計算

@timecounter
def mnist_test():

    # データセットを入手
    # 70000枚の28x28(784px)の手書き文字画像データ
    # 訓練用60000枚、テスト用10000枚に分割
    train, test = datasets.get_mnist()
    print type(train), type(test)
    print type(train[0][0]), type(train[0][1])
    print >> file(r'E:\log_data_trait.txt', 'w'), [train[i] for i in range(10)]
    print >> file(r'E:\log_data_test.txt', 'w'), [test[i] for i in range(10)]

    # 訓練データのイテレータ
    # ループ毎にシャッフルされる
    train_iter = iterators.SerialIterator(train, batch_size=100)

    # テストデータのイテレータ
    test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    # 多層パーセプトロンモデルを定義
    class MLP(Chain):

        def __init__(self):
            super(MLP, self).__init__(

                # Links
                l1=L.Linear(784, 100), # 入力層
                l2=L.Linear(100, 100), # 隠れ層
                l3=L.Linear(100, 10),  # 出力層
                )

        def __call__(self, x):

            # RELUを活性化関数として使う
            # ドロップアウトもここで指定
            h1 = F.dropout(F.relu(self.l1(x)))  # 入力層の出力
            h2 = F.dropout(F.relu(self.l2(h1))) # 隠れ層の出力
            #h1 = F.relu(self.l1(x))
            #h2 = F.relu(self.l2(h1))
            y = self.l3(h2)          # 出力層の出力
            return y

    model = L.Classifier(MLP(), lossfun=F.loss.softmax_cross_entropy.softmax_cross_entropy,
                         accfun=F.evaluation.accuracy.accuracy)

    # 最適化の設定
    optimizer = optimizers.SGD() # 確率的勾配降下法
    optimizer.setup(model)       # 最適化の準備
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005)) # ドロップコネクト

    updater = training.StandardUpdater(train_iter, optimizer)

    # モデルを訓練
    # 訓練を20エポック
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')

    # ログファイル
    log_name = r'E:\chainer\log.json'

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

    mnist_test()